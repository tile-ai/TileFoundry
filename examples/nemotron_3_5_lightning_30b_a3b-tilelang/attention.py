#!/usr/bin/env python
"""Nemotron's full-attention layer on its own, as a placement ladder.

The decode step's other two mixers do not grow with the context; this one does,
and it is the only part of the step whose best placement is different at the two
ends of the table. Pulling it out of `model.py` is what lets each placement be
analyzed on its own and the boundary between them be read off a number rather
than guessed -- the shape `tilefoundry tutorial authoring` lays out.

    by_head      one query head per unit, the whole context inside it
    by_context   one KV head per unit x a worker axis over the context
    by_both      both axes at once
    dispatch     the prototype that picks one by `ctx_full`

Every Module keeps one public signature: the pre-normed row in, the layer's mix
and this token's K/V rows out, which is exactly what one attention layer of the
mega step consumes and produces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tilefoundry import func, module
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare tile() in the scan
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

H, HQ, HKV, DH = 2688, 32, 2, 128
GQA = HQ // HKV
QP, KVP = HQ * DH, HKV * DH
CAP, ABLK = 262144, 128
QSCALE = DH ** -0.5
NEGINF = -1e30
_DT = "bf16"

#: The context arrives as two views of one buffer, the way `prepare_inputs_for_
#: generation` cuts it: whole ABLK blocks, then the positions that did not fill
#: one. `ctx_full` is what the dispatch keys on -- `ctx_tail` never exceeds ABLK.
CF = DimVar("ctx_full", 0, CAP + 1)
CT = DimVar("ctx_tail", 1, ABLK + 1)

#: Workers over the context, for the placements that carry that axis.
WRK = 32
WRK_BOTH = 4

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", 132)

#: Where a CTA stops being able to hold its own head's whole cache on chip. One
#: query head reads one KV head: K and V, 128 dimensions, two bytes each.
SMEM_BUDGET = 232448
CACHE_BYTES_PER_POSITION = 2 * HKV * DH * 2 // HKV
CAPACITY_T = SMEM_BUDGET // CACHE_BYTES_PER_POSITION


#: The ladder compares one decision, so the projections are not in it: they are
#: the same four matmuls whichever way the scan is placed, and left in they
#: dominate the per-unit flops and hide what is being compared. What is in it is
#: the scan -- q in, the context vector out, the cache read along the way.


@module(entry="scan", target=_H200, topologies=(_CTA,))
class ScanByHead:
    """One query head per unit; each unit walks the whole context itself.

    Nothing crosses units, so there is no worker axis and no log-sum-exp
    merge: a head is finished where it is computed. This is the placement
    worth having while walking the whole context is cheaper than carrying
    all 32 heads of a context-split unit."""

    @func
    def scan(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ):
        with Mesh(("cta",), layout=(HKV, GQA), names=("g", "q")) as kv:
            qs = tf.reshard(qg, (1, HKV @ kv.g, GQA @ kv.q, DH), "smem")
            m0 = tf.zeros(Tensor[(1, HKV @ kv.g, GQA @ kv.q, 1), "f32", "smem"])
            a0 = tf.zeros(Tensor[(1, HKV @ kv.g, GQA @ kv.q, DH), "f32", "smem"])
            m = tf.full_like(m0, value=NEGINF)
            l = tf.full_like(m0, value=0.0)
            acc = tf.full_like(a0, value=0.0)
            # Every block, in order, on this one unit: no worker axis, so no
            # log-sum-exp merge and no remainder that belongs to somebody else.
            for t in tile(CF, ABLK):
                b0 = t
                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                rw = tf.cast(tf.matmul(qs, kb, b_layout="NK"), dtype="f32")
                sb = rw * tf.full_like(rw, value=QSCALE)
                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind="max")
                nm = tf.max(m, bm)
                cr = tf.exp(m - nm)
                pw = tf.exp(sb - nm)
                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind="sum")
                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb), dtype="f32")
                m = nm
            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            rwt = tf.cast(tf.matmul(qs, kbt, b_layout="NK"), dtype="f32")
            sbt = rwt * tf.full_like(rwt, value=QSCALE)
            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind="max")
            nmt = tf.max(m, bmt)
            crt = tf.exp(m - nmt)
            pwt = tf.exp(sbt - nmt)
            lt = l * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind="sum")
            act = acc * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt), dtype="f32")
            ct = tf.cast(act / lt, dtype=_DT)
            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), "gmem"), new_shape=(1, QP))


@module(entry="scan", target=_H200, topologies=(_CTA,))
class ScanByContext:
    """One KV head per unit x a worker axis over the context.

    Past a few thousand positions this is the only split with units enough
    in it: the head axis is 32 units however long the context gets, while
    blocks of context are ceil(T/128) and fill the grid. The price is a
    log-sum-exp merge across the worker axis.

    The remainder block is merged once, into the already-merged state, and
    not inside the worker-sharded scan -- a worker-sharded merge of it puts
    the same positions in every worker's partial and the combine then adds
    them up WRK times."""

    @func
    def scan(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ):
        with Mesh(("cta",), layout=(HKV, WRK), names=("g", "w")) as kv:
            qs = tf.reshard(qg, (1, HKV @ kv.g, GQA, DH), "smem")
            m0 = tf.zeros(Tensor[(WRK @ kv.w, HKV @ kv.g, GQA, 1), "f32", "smem"])
            a0 = tf.zeros(Tensor[(WRK @ kv.w, HKV @ kv.g, GQA, DH), "f32", "smem"])
            m = tf.full_like(m0, value=NEGINF)
            l = tf.full_like(m0, value=0.0)
            acc = tf.full_like(a0, value=0.0)
            for t in tile(CF, ABLK * WRK):
                b0 = t + kv.w * ABLK
                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                rw = tf.cast(tf.matmul(qs, kb, b_layout="NK"), dtype="f32")
                sb = rw * tf.full_like(rw, value=QSCALE)
                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind="max")
                nm = tf.max(m, bm)
                cr = tf.exp(m - nm)
                pw = tf.exp(sb - nm)
                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind="sum")
                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb), dtype="f32")
                m = nm
            am = tf.reshard(m, (WRK, HKV @ kv.g, GQA, 1), "smem")
            al = tf.reshard(l, (WRK, HKV @ kv.g, GQA, 1), "smem")
            aa = tf.reshard(acc, (WRK, HKV @ kv.g, GQA, DH), "smem")
            gm = tf.reduce(am, axes=(0,), keepdim=True, kind="max")
            cw = tf.exp(am - gm)
            gl = tf.reduce(cw * al, axes=(0,), keepdim=True, kind="sum")
            ga = tf.reduce(cw * aa, axes=(0,), keepdim=True, kind="sum")
            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            rwt = tf.cast(tf.matmul(qs, kbt, b_layout="NK"), dtype="f32")
            sbt = rwt * tf.full_like(rwt, value=QSCALE)
            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind="max")
            nmt = tf.max(gm, bmt)
            crt = tf.exp(gm - nmt)
            pwt = tf.exp(sbt - nmt)
            lt = gl * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind="sum")
            act = ga * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt), dtype="f32")
            ct = tf.cast(act / lt, dtype=_DT)
            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), "gmem"), new_shape=(1, QP))


@module(entry="scan", target=_H200, topologies=(_CTA,))
class ScanBoth:
    """Both axes at once: one query head per unit and workers over the context.

    32 heads x 4 workers is 128 units against 64 for KV-head-and-worker,
    and it is the shape both ends of the table want -- the head axis carries
    the short end where there are few blocks to divide, the worker axis the
    long end. What it costs is reuse: a K element staged by a unit holding
    one query head is dotted once, where a unit holding all 16 heads of a
    group dots it 16 times."""

    @func
    def scan(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ):
        with Mesh(("cta",), layout=(HKV, GQA, WRK_BOTH), names=("g", "q", "w")) as kv:
            qs = tf.reshard(qg, (1, HKV @ kv.g, GQA @ kv.q, DH), "smem")
            m0 = tf.zeros(Tensor[(WRK_BOTH @ kv.w, HKV @ kv.g, GQA @ kv.q, 1), "f32", "smem"])
            a0 = tf.zeros(Tensor[(WRK_BOTH @ kv.w, HKV @ kv.g, GQA @ kv.q, DH), "f32", "smem"])
            m = tf.full_like(m0, value=NEGINF)
            l = tf.full_like(m0, value=0.0)
            acc = tf.full_like(a0, value=0.0)
            for t in tile(CF, ABLK * WRK_BOTH):
                b0 = t + kv.w * ABLK
                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                rw = tf.cast(tf.matmul(qs, kb, b_layout="NK"), dtype="f32")
                sb = rw * tf.full_like(rw, value=QSCALE)
                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind="max")
                nm = tf.max(m, bm)
                cr = tf.exp(m - nm)
                pw = tf.exp(sb - nm)
                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind="sum")
                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb), dtype="f32")
                m = nm
            am = tf.reshard(m, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, 1), "smem")
            al = tf.reshard(l, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, 1), "smem")
            aa = tf.reshard(acc, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, DH), "smem")
            gm = tf.reduce(am, axes=(0,), keepdim=True, kind="max")
            cw = tf.exp(am - gm)
            gl = tf.reduce(cw * al, axes=(0,), keepdim=True, kind="sum")
            ga = tf.reduce(cw * aa, axes=(0,), keepdim=True, kind="sum")
            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            rwt = tf.cast(tf.matmul(qs, kbt, b_layout="NK"), dtype="f32")
            sbt = rwt * tf.full_like(rwt, value=QSCALE)
            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind="max")
            nmt = tf.max(gm, bmt)
            crt = tf.exp(gm - nmt)
            pwt = tf.exp(sbt - nmt)
            lt = gl * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind="sum")
            act = ga * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt), dtype="f32")
            ct = tf.cast(act / lt, dtype=_DT)
            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), "gmem"), new_shape=(1, QP))


#: Where the two placements cross, read off the ladder rather than tuned: at
#: ctx_full = 2048 the per-unit work of walking the whole context with one query
#: head (1,051,265 bf16 flops) passes the per-unit work of carrying all 32 heads
#: over a stripe of it (1,060,880). Below it the head axis is the one with units
#: in it; above it the cache is too big to walk 16 times over.
CROSSOVER = 2048


@module(entry="attend", target=_H200, topologies=(_CTA,))
class AttnDispatch:
    """The scan, placed by how long the context is.

    The prototype carries the signature and the envelope; the two variants carry
    the placements. `variant_for(fn, {"ctx_full": n})` is what picks one, and the
    runtime twin keys its two kernel bodies on the same number, so the boundary
    is stated once and read twice.
    """

    @func
    def by_head_worker(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ):
        with Mesh(("cta",), layout=(HKV, GQA, WRK_BOTH), names=("g", "q", "w")) as kv:
            qs = tf.reshard(qg, (1, HKV @ kv.g, GQA @ kv.q, DH), "smem")
            m0 = tf.zeros(Tensor[(WRK_BOTH @ kv.w, HKV @ kv.g, GQA @ kv.q, 1), "f32", "smem"])
            a0 = tf.zeros(Tensor[(WRK_BOTH @ kv.w, HKV @ kv.g, GQA @ kv.q, DH), "f32", "smem"])
            m = tf.full_like(m0, value=NEGINF)
            l = tf.full_like(m0, value=0.0)
            acc = tf.full_like(a0, value=0.0)
            # Every WRK_BOTH-th block, so the four workers of a head between
            # them walk the context once rather than four times.
            for t in tile(CF, ABLK * WRK_BOTH):
                b0 = t + kv.w * ABLK
                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                rw = tf.cast(tf.matmul(qs, kb, b_layout="NK"), dtype="f32")
                sb = rw * tf.full_like(rw, value=QSCALE)
                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind="max")
                nm = tf.max(m, bm)
                cr = tf.exp(m - nm)
                pw = tf.exp(sb - nm)
                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind="sum")
                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb), dtype="f32")
                m = nm
            # The workers merge before the remainder, not after: merging it into
            # the worker-sharded state puts the same positions into every
            # worker's partial and the combine then adds them up four times.
            am = tf.reshard(m, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, 1), "smem")
            al = tf.reshard(l, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, 1), "smem")
            aa = tf.reshard(acc, (WRK_BOTH, HKV @ kv.g, GQA @ kv.q, DH), "smem")
            gm = tf.reduce(am, axes=(0,), keepdim=True, kind="max")
            cw = tf.exp(am - gm)
            gl = tf.reduce(cw * al, axes=(0,), keepdim=True, kind="sum")
            ga = tf.reduce(cw * aa, axes=(0,), keepdim=True, kind="sum")
            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            rwt = tf.cast(tf.matmul(qs, kbt, b_layout="NK"), dtype="f32")
            sbt = rwt * tf.full_like(rwt, value=QSCALE)
            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind="max")
            nmt = tf.max(gm, bmt)
            crt = tf.exp(gm - nmt)
            pwt = tf.exp(sbt - nmt)
            lt = gl * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind="sum")
            act = ga * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt), dtype="f32")
            ct = tf.cast(act / lt, dtype=_DT)
            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), "gmem"), new_shape=(1, QP))

    @func
    def by_context(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ):
        with Mesh(("cta",), layout=(HKV, WRK), names=("g", "w")) as kv:
            qs = tf.reshard(qg, (1, HKV @ kv.g, GQA, DH), "smem")
            m0 = tf.zeros(Tensor[(WRK @ kv.w, HKV @ kv.g, GQA, 1), "f32", "smem"])
            a0 = tf.zeros(Tensor[(WRK @ kv.w, HKV @ kv.g, GQA, DH), "f32", "smem"])
            m = tf.full_like(m0, value=NEGINF)
            l = tf.full_like(m0, value=0.0)
            acc = tf.full_like(a0, value=0.0)
            for t in tile(CF, ABLK * WRK):
                b0 = t + kv.w * ABLK
                kb = tf.reshard(tf.transpose(k_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                vb = tf.reshard(tf.transpose(v_cache[:, b0:b0 + ABLK, :, :], perm=(0, 2, 1, 3)),
                                (1, HKV @ kv.g, ABLK, DH), "smem")
                rw = tf.cast(tf.matmul(qs, kb, b_layout="NK"), dtype="f32")
                sb = rw * tf.full_like(rw, value=QSCALE)
                bm = tf.reduce(sb, axes=(-1,), keepdim=True, kind="max")
                nm = tf.max(m, bm)
                cr = tf.exp(m - nm)
                pw = tf.exp(sb - nm)
                l = l * cr + tf.reduce(pw, axes=(-1,), keepdim=True, kind="sum")
                acc = acc * cr + tf.cast(tf.matmul(tf.cast(pw, dtype=_DT), vb), dtype="f32")
                m = nm
            am = tf.reshard(m, (WRK, HKV @ kv.g, GQA, 1), "smem")
            al = tf.reshard(l, (WRK, HKV @ kv.g, GQA, 1), "smem")
            aa = tf.reshard(acc, (WRK, HKV @ kv.g, GQA, DH), "smem")
            gm = tf.reduce(am, axes=(0,), keepdim=True, kind="max")
            cw = tf.exp(am - gm)
            gl = tf.reduce(cw * al, axes=(0,), keepdim=True, kind="sum")
            ga = tf.reduce(cw * aa, axes=(0,), keepdim=True, kind="sum")
            kbt = tf.reshard(tf.transpose(k_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            vbt = tf.reshard(tf.transpose(v_tail[:, 0:CT, :, :], perm=(0, 2, 1, 3)),
                             (1, HKV @ kv.g, CT, DH), "smem")
            rwt = tf.cast(tf.matmul(qs, kbt, b_layout="NK"), dtype="f32")
            sbt = rwt * tf.full_like(rwt, value=QSCALE)
            bmt = tf.reduce(sbt, axes=(-1,), keepdim=True, kind="max")
            nmt = tf.max(gm, bmt)
            crt = tf.exp(gm - nmt)
            pwt = tf.exp(sbt - nmt)
            lt = gl * crt + tf.reduce(pwt, axes=(-1,), keepdim=True, kind="sum")
            act = ga * crt + tf.cast(tf.matmul(tf.cast(pwt, dtype=_DT), vbt), dtype="f32")
            ct = tf.cast(act / lt, dtype=_DT)
            return tf.reshape(tf.reshard(ct, (1, HKV, GQA, DH), "gmem"), new_shape=(1, QP))

    @func
    def attend(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ) -> Tensor[(1, QP), _DT]:
        pass

    @attend.specialize(DimVarRangePat("ctx_full", 0, CROSSOVER))
    def attend_short(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ) -> Tensor[(1, QP), _DT]:
        return by_head_worker(qg, k_cache, v_cache, k_tail, v_tail)

    @attend.specialize(DimVarRangePat("ctx_full", CROSSOVER, CAP + 1))
    def attend_long(
        qg: Tensor[(1, HKV, GQA, DH), _DT],
        k_cache: Tensor[(1, CF, HKV, DH), _DT],
        v_cache: Tensor[(1, CF, HKV, DH), _DT],
        k_tail: Tensor[(1, CT, HKV, DH), _DT],
        v_tail: Tensor[(1, CT, HKV, DH), _DT],
    ) -> Tensor[(1, QP), _DT]:
        return by_context(qg, k_cache, v_cache, k_tail, v_tail)


@runtime_module(AttnDispatch)
class AttnRuntime:
    """The kernel's two attention bodies, against the ladder that states them.

    The step inlines these -- a Call from one HIR Function to another is a device
    call inside the same launch -- so `mega_kernel` emits the same two bodies a
    second time around a pair of parameters, and this is what puts them under
    `check`. Checking the same three functions on the twin of the whole model is
    what one would rather do, and cannot: drawing inputs for one leaf of that
    module materialises all 474 of its declared weights on both sides, which is
    139 GB (see ISSUES.md).
    """

    @runtime_func
    def by_head_worker(self, qg, k_cache, v_cache, k_tail, v_tail):
        import mega_kernel  # noqa: PLC0415 -- compiled on first use

        return mega_kernel.run_attn("head", qg, k_cache, v_cache, k_tail, v_tail)

    @runtime_func
    def by_context(self, qg, k_cache, v_cache, k_tail, v_tail):
        import mega_kernel  # noqa: PLC0415

        return mega_kernel.run_attn("context", qg, k_cache, v_cache, k_tail, v_tail)

    @runtime_func
    def attend(self, qg, k_cache, v_cache, k_tail, v_tail):
        import mega_kernel  # noqa: PLC0415

        return mega_kernel.run_attn("dispatch", qg, k_cache, v_cache, k_tail, v_tail)


#: The ladder's placements, in the order the boundary question reads them.
LADDER = {"by_head": ScanByHead, "by_context": ScanByContext, "by_both": ScanBoth}
_UNITS = {"by_head": HKV * GQA, "by_context": HKV * WRK, "by_both": HKV * GQA * WRK_BOTH}
_HEAD_RE = None


def measure(mod, full, tail):
    """The header numbers of one placement at one context length."""
    import re  # noqa: PLC0415
    from tilefoundry.analysis import analyze as run_analysis  # noqa: PLC0415
    from tilefoundry.inspection.analysis_report import (  # noqa: PLC0415
        render_analysis, render_text,
    )

    res = run_analysis(mod, mod.entry_function(), analysis=["roofline", "compute-cost",
                                                           "memory"],
                       dims={"ctx_full": full, "ctx_tail": tail})
    text = render_text(render_analysis(res))
    got = {}
    m = re.search(r"flops=bf16:(\d+)@(\d+)", text)
    if m:
        got["flops"], got["flops_per_unit"] = int(m[1]), int(m[2])
    m = re.search(r"gmem:r(\d+)/w(\d+)@r(\d+)/w(\d+)", text)
    if m:
        got["gmem_r"], got["gmem_r_per_unit"] = int(m[1]), int(m[3])
    m = re.search(r"smem:r(\d+)/w(\d+)@r(\d+)/w(\d+)", text)
    if m:
        got["smem_r_per_unit"] = int(m[3])
    m = re.search(r"peak-footprint=gmem:(\d+).*?smem:(\d+)", text)
    if m:
        got["smem_peak"] = int(m[2])
    m = re.search(r"ideal-ns=(\d+)", text)
    if m:
        got["ideal_ns"] = int(m[1])
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", default="0:1,128:1,1024:1,4096:1,32768:1,262016:128")
    ap.add_argument("--out", default="reports/attn_ladder.json")
    a = ap.parse_args()
    pts = [tuple(int(x) for x in p.split(":")) for p in a.points.split(",")]
    rows = []
    print(f"{'placement':<11}{'units':>6}{'ctx':>9}{'ideal ns':>10}"
          f"{'flops/unit':>12}{'gmem r/unit':>13}{'smem r/unit':>13}{'smem peak':>11}")
    for name, mod in LADDER.items():
        for full, tail in pts:
            try:
                got = measure(mod, full, tail)
            except Exception as error:  # noqa: BLE001 -- a refusal is a result
                got = {"error": f"{type(error).__name__}: {error}"[:120]}
            rows.append({"placement": name, "units": _UNITS[name],
                         "ctx_full": full, "ctx_tail": tail, **got})
            if "error" in got:
                print(f"{name:<11}{_UNITS[name]:>6}{full + tail:>9}  {got['error']}")
            else:
                print(f"{name:<11}{_UNITS[name]:>6}{full + tail:>9}{got['ideal_ns']:>10}"
                      f"{got['flops_per_unit']:>12}{got['gmem_r_per_unit']:>13}"
                      f"{got.get('smem_r_per_unit', 0):>13}{got.get('smem_peak', 0):>11}",
                      flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
