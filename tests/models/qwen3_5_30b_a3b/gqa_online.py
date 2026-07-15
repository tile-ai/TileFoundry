"""Flash / online-softmax GQA decode core + context-length `specialize`, with
the two compute-distribution strategies expressed in the dataflow itself.

Decode regime: batch 1, a small dynamic query length ``S`` (``seq_len``, designed
range 1..4 for speculative / chunked decode), ``NUM_Q_HEADS`` query heads sharing
``NUM_KV_HEADS`` KV heads (``GQA_GROUP`` queries per KV head), head dim
``HEAD_DIM``. The KV cache / active context length ``C`` (``ctx_len``) is the
large dynamic dimension (designed up to ``MAX_CTX`` = 256K) — and the dimension
the GQA prototype specializes on.

Authoring surface (per docs/spec/parser.md §8): ``gqa_online_attend`` is a
``pass``-bodied dispatch **prototype**; the two strategies are registered as
``@gqa_online_attend.specialize(...)`` variants, selected by the runtime
``ctx_len``:

- small context → **head-on-CTA**: query heads are the parallel axis (each CTA
  owns a head subset and runs the *full* online-softmax over the context — heads
  are embarrassingly parallel, no cross-CTA combine). The strategy is marked by
  a ``ShardLayout`` that ``Split``s the query-head axis across the CTA mesh.
- large context → **context-on-CTA (split-KV)**: two kernels written as two
  module-level ``@func``s. ``_ctx_partials`` reshapes the KV context into
  ``NUM_SPLITS`` contiguous blocks (one per CTA, ``NUM_SPLITS == NUM_CTA``) and
  computes each block's *own* local softmax partial ``(m_p, l_p, o_p)``;
  ``_ctx_combine`` flash-merges the partials over the ``NUM_SPLITS`` axis. The
  function-call boundary *models* the intended kernel boundary for lowering —
  the cross-CTA handoff (materialize partials to gmem, or a persistent-kernel
  barrier) is lowering work, not expressed in HIR and not realized by this
  example. For ``C % NUM_SPLITS != 0`` the block
  reshape size-mismatches and raises — the variant **fails closed** rather than
  silently dropping the tail and returning a wrong answer.

``DimVar`` bounds are half-open ``[lo, hi)`` (``hi`` exclusive; see the
partition verifier): the envelope upper bounds below are ``MAX_* + 1`` so the
maximum supported length is included.
"""
from __future__ import annotations

import math

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by the @func body
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op names for the @func body
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

# Model dims (match tests/models/qwen3_5_30b_a3b/common.py).
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 8

# Decode regime ceilings (largest supported lengths, inclusive value). DimVar /
# DimVarRangePat bounds are half-open [lo, hi), so the envelopes below use
# MAX_* + 1 as the exclusive upper bound.
MAX_SEQ = 4            # query length 1..4
MAX_CTX = 262144       # context / cache length 1..262144 (256K)

# Context-length boundary that selects the distribution strategy (NOT a
# performance threshold): head-on-CTA covers [1, SMALL_CONTEXT_T) (ctx_len up
# to SMALL_CONTEXT_T - 1); context-on-CTA covers [SMALL_CONTEXT_T, MAX_CTX + 1).
SMALL_CONTEXT_T = 4096

# CTA mesh extent; the context-on-CTA strategy cuts the KV context into this many
# contiguous blocks, intended to map one block per CTA at lowering. 256K is
# divisible by it.
NUM_CTA = 8
NUM_SPLITS = NUM_CTA

S = DimVar("seq_len", 1, MAX_SEQ + 1)   # [1, 5) = 1..4
C = DimVar("ctx_len", 1, MAX_CTX + 1)   # [1, 262145) = 1..262144
# Per-split context block (context-on-CTA): C is cut into NUM_SPLITS blocks, so
# the block length is ctx_len / NUM_SPLITS. A named DimVar lets it appear in the
# reshape target (a bare dim arithmetic expression cannot); the concrete extent
# is inferred from the runtime tensor at reshape.
CBLK = DimVar("ctx_blk", 1, MAX_CTX // NUM_SPLITS + 1)

_D = HEAD_DIM
_HQ = NUM_Q_HEADS
_HKV = NUM_KV_HEADS
_G = GQA_GROUP
_SCALE = 1.0 / math.sqrt(HEAD_DIM)


@func(topologies=(Topology("cta", NUM_CTA),))
def gqa_online_attend(
    q: Tensor[(1, S, _HQ, _D), "bf16"],
    k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
    v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
) -> Tensor[(1, S, _HQ, _D), "bf16"]:
    # Dispatch prototype: the strategy implementations live in the two
    # `.specialize` variants below, selected by the runtime ctx_len.
    pass


@gqa_online_attend.specialize(DimVarRangePat("ctx_len", 1, SMALL_CONTEXT_T))
def _(
    q: Tensor[(1, S, _HQ, _D), "bf16"],
    k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
    v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
) -> Tensor[(1, S, _HQ, _D), "bf16"]:
    # head-on-CTA: split the query-head axis across the CTA mesh (`_HQ @ cta`),
    # each CTA running the full online-softmax over the context (heads are
    # embarrassingly parallel — no cross-CTA combine). All mesh-scoped compute
    # stays inside the `with Mesh` block.
    with Mesh(topology="cta", layout=Layout((NUM_CTA,), (1,))) as cta:
        q_sh = reshard(q, layout=(1, S, _HQ @ cta, _D))  # reshard + cta layout sugar
        q_f = tf.cast(q_sh, dtype="f32")
        q_s = q_f * tf.full_like(q_f, value=_SCALE)
        tmpl = tf.reduce(q_f, axes=(-1,), keepdim=True, kind="sum")  # [1, S, Hq, 1]
        m = tf.full_like(tmpl, value=-1e30)
        l = tf.full_like(tmpl, value=0.0)
        o = tf.full_like(q_f, value=0.0)
        for i in tile(C):  # scan the full context locally
            k_i = tf.reshape(
                tf.cast(tf.repeat_interleave(tf.gather(k_cache, i, axis=1), repeats=_G, axis=1), dtype="f32"),
                new_shape=(1, 1, _HQ, _D),
            )
            v_i = tf.reshape(
                tf.cast(tf.repeat_interleave(tf.gather(v_cache, i, axis=1), repeats=_G, axis=1), dtype="f32"),
                new_shape=(1, 1, _HQ, _D),
            )
            score = tf.reduce(q_s * k_i, axes=(-1,), keepdim=True, kind="sum")
            m_new = tf.max(m, score)
            p = tf.exp(score - m_new)
            corr = tf.exp(m - m_new)
            l = l * corr + p
            o = o * corr + p * v_i
            m = m_new
        return tf.cast(o / l, dtype="bf16")


# context-on-CTA (split-KV) is two kernels: a per-CTA `partials` pass over each
# context block, then a `combine` pass that flash-merges the partials. The two
# module-level @funcs make the intended kernel boundary — and the cross-CTA
# handoff it implies (a lowering concern, not realized here) — explicit, instead
# of hiding the combine inside one function.


@func(topologies=(Topology("cta", NUM_CTA),))
def _ctx_partials(
    q: Tensor[(1, S, _HQ, _D), "bf16"],
    k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
    v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
):
    # Per-CTA partial. Reshape the KV context into NUM_SPLITS contiguous blocks
    # (block length CBLK inferred from runtime ctx_len; a non-aligned ctx_len
    # size-mismatches the reshape and fails closed), one block per CTA. Each
    # block gets its OWN local softmax (m_p, l_p, o_p) over its block axis only —
    # no cross-block reduction here. `with Mesh(cta)` is the CTA-parallel
    # authoring context (NUM_SPLITS is the CTA axis); it emits no IR node.
    with Mesh(topology="cta", layout=Layout((NUM_CTA,), (1,))) as cta:  # noqa: F841
        k_f = tf.transpose(
            tf.cast(tf.repeat_interleave(
                tf.reshape(k_cache, new_shape=(1, NUM_SPLITS, CBLK, _HKV, _D)),
                repeats=_G, axis=3), dtype="f32"),
            perm=(0, 3, 1, 2, 4),
        )  # [1, Hq, NUM_SPLITS, CBLK, D]
        v_f = tf.transpose(
            tf.cast(tf.repeat_interleave(
                tf.reshape(v_cache, new_shape=(1, NUM_SPLITS, CBLK, _HKV, _D)),
                repeats=_G, axis=3), dtype="f32"),
            perm=(0, 3, 1, 2, 4),
        )  # [1, Hq, NUM_SPLITS, CBLK, D]
        q_f = tf.cast(q, dtype="f32")
        q_s = q_f * tf.full_like(q_f, value=_SCALE)  # [1, S, Hq, D]
        # scores[1, S, Hq, NUM_SPLITS, CBLK, 1] = Σ_D q·k (head dim kept as a unit
        # slot so the value mix broadcasts without a 2-dynamic-axis reshape).
        q_e = tf.reshape(q_s, new_shape=(1, S, _HQ, 1, 1, _D))
        k_e = tf.reshape(k_f, new_shape=(1, 1, _HQ, NUM_SPLITS, CBLK, _D))
        v_e = tf.reshape(v_f, new_shape=(1, 1, _HQ, NUM_SPLITS, CBLK, _D))
        scores = tf.reduce(q_e * k_e, axes=(-1,), keepdim=True, kind="sum")
        m_p = tf.reduce(scores, axes=(-2,), keepdim=True, kind="max")  # max over block
        p = tf.exp(scores - m_p)
        l_p = tf.reduce(p, axes=(-2,), keepdim=True, kind="sum")
        o_p = tf.reduce(p * v_e, axes=(-2,), keepdim=False, kind="sum")  # [1,S,Hq,NS,D]
        return (
            tf.reshape(m_p, new_shape=(1, S, _HQ, NUM_SPLITS, 1)),
            tf.reshape(l_p, new_shape=(1, S, _HQ, NUM_SPLITS, 1)),
            o_p,
        )


@func(topologies=(Topology("cta", NUM_CTA),))
def _ctx_combine(
    m_p: Tensor[(1, S, _HQ, NUM_SPLITS, 1), "f32"],
    l_p: Tensor[(1, S, _HQ, NUM_SPLITS, 1), "f32"],
    o_p: Tensor[(1, S, _HQ, NUM_SPLITS, _D), "f32"],
) -> Tensor[(1, S, _HQ, _D), "bf16"]:
    # Flash log-sum-exp merge of the per-CTA partials over the NUM_SPLITS axis.
    # `with Mesh(cta)` keeps this on the same CTA mesh as `_ctx_partials`: the
    # NUM_SPLITS axis IS the CTA axis, so this reduction is the cross-CTA combine,
    # mapped at lowering to a barrier / gather over the producers (not an IR node).
    with Mesh(topology="cta", layout=Layout((NUM_CTA,), (1,))) as cta:  # noqa: F841
        m = tf.reduce(m_p, axes=(-2,), keepdim=True, kind="max")          # global max
        alpha = tf.exp(m_p - m)                                           # rescale per block
        l = tf.reduce(alpha * l_p, axes=(-2,), keepdim=True, kind="sum")  # [1,S,Hq,1,1]
        o = tf.reduce(alpha * o_p, axes=(-2,), keepdim=False, kind="sum") # [1,S,Hq,D]
        return tf.cast(o / tf.reshape(l, new_shape=(1, S, _HQ, 1)), dtype="bf16")


@gqa_online_attend.specialize(DimVarRangePat("ctx_len", SMALL_CONTEXT_T, MAX_CTX + 1))
def _(
    q: Tensor[(1, S, _HQ, _D), "bf16"],
    k_cache: Tensor[(1, C, _HKV, _D), "bf16"],
    v_cache: Tensor[(1, C, _HKV, _D), "bf16"],
) -> Tensor[(1, S, _HQ, _D), "bf16"]:
    # Two kernels: per-CTA partials, then a flash combine. The function-call
    # boundary models the intended kernel boundary; lowering (not this example)
    # realizes the cross-CTA handoff.
    m_p, l_p, o_p = _ctx_partials(q, k_cache, v_cache)
    return _ctx_combine(m_p, l_p, o_p)


__all__ = [
    "gqa_online_attend",
    "SMALL_CONTEXT_T",
    "NUM_CTA",
    "NUM_SPLITS",
    "MAX_SEQ",
    "MAX_CTX",
    "HEAD_DIM",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "GQA_GROUP",
]
