#!/usr/bin/env python3
"""A small GQA decode attention ladder for the authoring tutorial.

The six Modules keep one public shape and change one placement decision at a
time.  The dimensions are intentionally small, but the query/KV ratio matches
the GQA shape used by the published Qwen attention model.
"""

from __future__ import annotations

import math
import re
import sys

from tilefoundry import func, module
from tilefoundry.analysis import analyze as run_analysis
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 - bare tile() in the fused body
from tilefoundry.inspection.analysis_report import render_analysis, render_text
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

HIDDEN = 256
QUERY_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 32
KV_DIM = KV_HEADS * HEAD_DIM
GQA_GROUP = QUERY_HEADS // KV_HEADS
ROPE_CONTEXT = 8192
CTX = DimVar("ctx_len", 1, ROPE_CONTEXT + 1)
SCALE = 1.0 / math.sqrt(HEAD_DIM)
WORKERS = 4
BLOCK = 128

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", 132)


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage0_Naive:
    """One unsplit CTA reads the complete attention sublayer."""

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        q = tf.reshape(tf.matmul(hidden, w_q), new_shape=(1, 1, QUERY_HEADS, HEAD_DIM))
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
        k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
        v_all = tf.cache_update(v_cache, cur_pos, write_len, v)

        k_heads = tf.repeat_interleave(k_all, repeats=GQA_GROUP, axis=2)
        v_heads = tf.repeat_interleave(v_all, repeats=GQA_GROUP, axis=2)
        k_heads = tf.transpose(k_heads, perm=(0, 2, 1, 3))
        v_heads = tf.transpose(v_heads, perm=(0, 2, 1, 3))
        q_f32 = tf.cast(q_rope, dtype="f32")
        k_f32 = tf.cast(k_heads, dtype="f32")
        v_f32 = tf.cast(v_heads, dtype="f32")
        scaled_q = q_f32 * tf.full_like(q_f32, value=SCALE)
        q_e = tf.reshape(scaled_q, new_shape=(1, 1, QUERY_HEADS, 1, HEAD_DIM))
        k_e = tf.reshape(k_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM))
        v_e = tf.reshape(v_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM))
        scores = tf.reduce(q_e * k_e, axes=(-1,), keepdim=True, kind="sum")
        peak = tf.reduce(scores, axes=(-2,), keepdim=True, kind="max")
        weights = tf.exp(scores - peak)
        normalizer = tf.reduce(weights, axes=(-2,), keepdim=False, kind="sum")
        weighted = tf.reduce(weights * v_e, axes=(-2,), keepdim=False, kind="sum")
        attended = weighted / normalizer
        attended_bf16 = tf.cast(attended, dtype="bf16")
        return tf.matmul(tf.reshape(attended_bf16, new_shape=(1, 1, HIDDEN)), w_o)


gqa_decode = Stage0_Naive.entry_function()


SMEM_BUDGET = 232448
CACHE_BYTES_PER_CONTEXT_PER_CTA = 2 * HEAD_DIM * 2
SPECIALIZE_T = SMEM_BUDGET // CACHE_BYTES_PER_CONTEXT_PER_CTA


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage1_Specialized:
    """Dispatch the same unsplit kernel at the measured context boundary."""

    @func
    def _decode_core(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        q = tf.reshape(tf.matmul(hidden, w_q), new_shape=(1, 1, QUERY_HEADS, HEAD_DIM))
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
        k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
        v_all = tf.cache_update(v_cache, cur_pos, write_len, v)
        k_heads = tf.transpose(
            tf.repeat_interleave(k_all, repeats=GQA_GROUP, axis=2), perm=(0, 2, 1, 3)
        )
        v_heads = tf.transpose(
            tf.repeat_interleave(v_all, repeats=GQA_GROUP, axis=2), perm=(0, 2, 1, 3)
        )
        q_f32 = tf.cast(q_rope, dtype="f32")
        k_f32 = tf.cast(k_heads, dtype="f32")
        v_f32 = tf.cast(v_heads, dtype="f32")
        scaled_q = q_f32 * tf.full_like(q_f32, value=SCALE)
        q_e = tf.reshape(scaled_q, new_shape=(1, 1, QUERY_HEADS, 1, HEAD_DIM))
        k_e = tf.reshape(k_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM))
        v_e = tf.reshape(v_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM))
        scores = tf.reduce(q_e * k_e, axes=(-1,), keepdim=True, kind="sum")
        peak = tf.reduce(scores, axes=(-2,), keepdim=True, kind="max")
        weights = tf.exp(scores - peak)
        normalizer = tf.reduce(weights, axes=(-2,), keepdim=False, kind="sum")
        weighted = tf.reduce(weights * v_e, axes=(-2,), keepdim=False, kind="sum")
        attended = tf.cast(weighted / normalizer, dtype="bf16")
        return tf.matmul(tf.reshape(attended, new_shape=(1, 1, HIDDEN)), w_o)

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        pass

    @gqa_decode.specialize(DimVarRangePat("ctx_len", 1, SPECIALIZE_T))
    def short_context(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        return _decode_core(
            hidden, w_q, w_k, w_v, w_o, k_cache, v_cache,
            cur_pos, write_len, pos_ids, cos_cache, sin_cache,
        )

    @gqa_decode.specialize(DimVarRangePat("ctx_len", SPECIALIZE_T, ROPE_CONTEXT + 1))
    def long_context(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        return _decode_core(
            hidden, w_q, w_k, w_v, w_o, k_cache, v_cache,
            cur_pos, write_len, pos_ids, cos_cache, sin_cache,
        )


gqa_decode_specialized = Stage1_Specialized.entry_function()


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage2_Sharded:
    """Give each CTA one query-head slice while keeping the cache whole."""

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        with Mesh(("cta",), layout=(QUERY_HEADS,), names=("head",)) as cta:
            q = tf.reshape(
                tf.matmul(hidden, w_q), new_shape=(1, 1, QUERY_HEADS, HEAD_DIM)
            )
            k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
            v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
            q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
            k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
            v_all = tf.cache_update(v_cache, cur_pos, write_len, v)
            q_sh = tf.reshard(
                q_rope, (1, 1, QUERY_HEADS @ cta.head, HEAD_DIM), "smem"
            )
            k_heads = tf.transpose(
                tf.repeat_interleave(k_all, repeats=GQA_GROUP, axis=2), perm=(0, 2, 1, 3)
            )
            v_heads = tf.transpose(
                tf.repeat_interleave(v_all, repeats=GQA_GROUP, axis=2), perm=(0, 2, 1, 3)
            )
            k_sh = tf.reshard(
                k_heads, (1, QUERY_HEADS @ cta.head, CTX, HEAD_DIM), "smem"
            )
            v_sh = tf.reshard(
                v_heads, (1, QUERY_HEADS @ cta.head, CTX, HEAD_DIM), "smem"
            )
            queries = tf.transpose(tf.cast(q_sh, dtype="f32"), perm=(0, 2, 1, 3))
            keys = tf.transpose(tf.cast(k_sh, dtype="f32"), perm=(0, 1, 3, 2))
            values = tf.transpose(tf.cast(v_sh, dtype="f32"), perm=(0, 1, 2, 3))
            scaled_q = queries * tf.full_like(queries, value=SCALE)
            scores = tf.matmul(scaled_q, keys)
            peak = tf.reduce(scores, axes=(-1,), keepdim=True, kind="max")
            weights = tf.exp(scores - peak)
            normalizer = tf.reduce(weights, axes=(-1,), keepdim=True, kind="sum")
            weighted = tf.matmul(weights, values)
            attended = tf.transpose(
                tf.cast(weighted / normalizer, dtype="bf16"), perm=(0, 2, 1, 3)
            )
            attended = tf.reshard(attended, (1, 1, QUERY_HEADS, HEAD_DIM), "gmem")
            return tf.matmul(tf.reshape(attended, new_shape=(1, 1, HIDDEN)), w_o)


gqa_decode_sharded = Stage2_Sharded.entry_function()


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage3_Fused:
    """Split the cache scan across workers and combine online-softmax partials."""

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        q = tf.reshape(
            tf.matmul(hidden, w_q), new_shape=(1, 1, QUERY_HEADS, HEAD_DIM)
        )
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
        k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
        v_all = tf.cache_update(v_cache, cur_pos, write_len, v)
        k_heads = tf.repeat_interleave(k_all, repeats=GQA_GROUP, axis=2)
        v_heads = tf.repeat_interleave(v_all, repeats=GQA_GROUP, axis=2)

        with Mesh(
            ("cta",), layout=(QUERY_HEADS, WORKERS), names=("head", "worker")
        ) as cta:
            qh = tf.reshard(
                q_rope, (1, 1, QUERY_HEADS @ cta.head, HEAD_DIM), "smem"
            )
            queries = tf.transpose(tf.cast(qh, dtype="f32"), perm=(0, 2, 1, 3))
            scaled = queries * tf.full_like(queries, value=SCALE)
            m_slots = tf.zeros(
                Tensor[
                    (WORKERS @ cta.worker, QUERY_HEADS @ cta.head, 1, 1),
                    "f32",
                    "smem",
                ]
            )
            acc_slots = tf.zeros(
                Tensor[
                    (WORKERS @ cta.worker, QUERY_HEADS @ cta.head, 1, HEAD_DIM),
                    "f32",
                    "smem",
                ]
            )
            m = tf.full_like(m_slots, value=-1e30)
            l = tf.full_like(m_slots, value=0.0)
            acc = tf.full_like(acc_slots, value=0.0)

            for start in tile(CTX, BLOCK * WORKERS):
                base = start + cta.worker * BLOCK
                kb = tf.reshard(
                    k_heads[:, base : base + BLOCK, :, :],
                    (1, BLOCK, QUERY_HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                vb = tf.reshard(
                    v_heads[:, base : base + BLOCK, :, :],
                    (1, BLOCK, QUERY_HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                keys = tf.transpose(tf.cast(kb, dtype="f32"), perm=(0, 2, 3, 1))
                values = tf.transpose(tf.cast(vb, dtype="f32"), perm=(0, 2, 1, 3))
                scores = tf.matmul(scaled, keys)
                block_m = tf.reduce(scores, axes=(-1,), keepdim=True, kind="max")
                next_m = tf.max(m, block_m)
                correction = tf.exp(m - next_m)
                weights = tf.exp(scores - next_m)
                l = l * correction + tf.reduce(
                    weights, axes=(-1,), keepdim=True, kind="sum"
                )
                acc = acc * correction + tf.matmul(weights, values)
                m = next_m

            all_m = tf.reshard(
                m, (WORKERS, QUERY_HEADS @ cta.head, 1, 1), "smem"
            )
            all_l = tf.reshard(
                l, (WORKERS, QUERY_HEADS @ cta.head, 1, 1), "smem"
            )
            all_acc = tf.reshard(
                acc, (WORKERS, QUERY_HEADS @ cta.head, 1, HEAD_DIM), "smem"
            )
            global_m = tf.reduce(all_m, axes=(0,), keepdim=False, kind="max")
            weights = tf.exp(all_m - global_m)
            global_l = tf.reduce(weights * all_l, axes=(0,), keepdim=False, kind="sum")
            global_acc = tf.reduce(
                weights * all_acc, axes=(0,), keepdim=False, kind="sum"
            )
            attended = tf.cast(global_acc / global_l, dtype="bf16")
            attended = tf.reshape(
                attended, new_shape=(1, 1, QUERY_HEADS, HEAD_DIM)
            )
            attended = tf.reshard(
                attended, (1, 1, QUERY_HEADS, HEAD_DIM), "gmem"
            )
            return tf.matmul(tf.reshape(attended, new_shape=(1, 1, HIDDEN)), w_o)


gqa_decode_fused = Stage3_Fused.entry_function()


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage4_WeightPrepared:
    """Stage projection weights by output slice before the attention scan."""

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        with Mesh(("cta",), layout=(QUERY_HEADS,), names=("head",)) as cta:
            wq_local = tf.reshard(
                w_q, (1, HIDDEN, HIDDEN @ cta.head), "smem"
            )
            wk_local = tf.reshard(
                w_k, (1, HIDDEN, KV_DIM @ cta.head), "smem"
            )
            wv_local = tf.reshard(
                w_v, (1, HIDDEN, KV_DIM @ cta.head), "smem"
            )
            hidden_local = tf.reshard(hidden, (1, 1, HIDDEN), "smem")
            q_projected = tf.matmul(hidden_local, wq_local)
            k_projected = tf.matmul(hidden_local, wk_local)
            v_projected = tf.matmul(hidden_local, wv_local)
            q_projected = tf.reshard(q_projected, (1, 1, HIDDEN), "gmem")
            k_projected = tf.reshard(k_projected, (1, 1, KV_DIM), "gmem")
            v_projected = tf.reshard(v_projected, (1, 1, KV_DIM), "gmem")
            q = tf.reshape(
                q_projected, new_shape=(1, 1, QUERY_HEADS, HEAD_DIM)
            )
            k = tf.reshape(k_projected, new_shape=(1, 1, KV_HEADS, HEAD_DIM))
            v = tf.reshape(v_projected, new_shape=(1, 1, KV_HEADS, HEAD_DIM))
            q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
            k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
            v_all = tf.cache_update(v_cache, cur_pos, write_len, v)
            k_heads = tf.transpose(
                tf.repeat_interleave(k_all, repeats=GQA_GROUP, axis=2),
                perm=(0, 2, 1, 3),
            )
            v_heads = tf.transpose(
                tf.repeat_interleave(v_all, repeats=GQA_GROUP, axis=2),
                perm=(0, 2, 1, 3),
            )
            q_f32 = tf.cast(q_rope, dtype="f32")
            k_f32 = tf.cast(k_heads, dtype="f32")
            v_f32 = tf.cast(v_heads, dtype="f32")
            scaled_q = q_f32 * tf.full_like(q_f32, value=SCALE)
            q_e = tf.reshape(
                scaled_q, new_shape=(1, 1, QUERY_HEADS, 1, HEAD_DIM)
            )
            k_e = tf.reshape(
                k_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM)
            )
            v_e = tf.reshape(
                v_f32, new_shape=(1, 1, QUERY_HEADS, CTX, HEAD_DIM)
            )
            scores = tf.reduce(q_e * k_e, axes=(-1,), keepdim=True, kind="sum")
            peak = tf.reduce(scores, axes=(-2,), keepdim=True, kind="max")
            weights = tf.exp(scores - peak)
            normalizer = tf.reduce(weights, axes=(-2,), keepdim=False, kind="sum")
            weighted = tf.reduce(weights * v_e, axes=(-2,), keepdim=False, kind="sum")
            attended = tf.cast(weighted / normalizer, dtype="bf16")
            w_o_local = tf.reshard(
                w_o, (1, HIDDEN, HIDDEN @ cta.head), "smem"
            )
            attended_local = tf.reshard(
                tf.reshape(attended, new_shape=(1, 1, HIDDEN)),
                (1, 1, HIDDEN),
                "smem",
            )
            output_local = tf.matmul(
                attended_local, w_o_local
            )
            return tf.reshard(output_local, (1, 1, HIDDEN), "gmem")


gqa_decode_weight_prepared = Stage4_WeightPrepared.entry_function()


@module(entry="gqa_decode", target=_H200, topologies=(_CTA,))
class Stage5_CachePrepared:
    """Stream cache blocks through smem while an online state stays resident."""

    @func
    def gqa_decode(
        hidden: Tensor[(1, 1, HIDDEN), "bf16"],
        w_q: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        w_k: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_v: ConstTensor[(1, HIDDEN, KV_DIM), "bf16"],
        w_o: ConstTensor[(1, HIDDEN, HIDDEN), "bf16"],
        k_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CTX, KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        write_len: Tensor[(1,), "i32"],
        pos_ids: Tensor[(1,), "i32"],
        cos_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(ROPE_CONTEXT, HEAD_DIM), "bf16"],
    ) -> Tensor[(1, 1, HIDDEN), "bf16"]:
        q = tf.reshape(
            tf.matmul(hidden, w_q), new_shape=(1, 1, QUERY_HEADS, HEAD_DIM)
        )
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, 1, KV_HEADS, HEAD_DIM))
        q_rope, k_rope = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
        k_all = tf.cache_update(k_cache, cur_pos, write_len, k_rope)
        v_all = tf.cache_update(v_cache, cur_pos, write_len, v)

        with Mesh(("cta",), layout=(QUERY_HEADS,), names=("head",)) as cta:
            qh = tf.reshard(
                q_rope, (1, 1, QUERY_HEADS @ cta.head, HEAD_DIM), "smem"
            )
            queries = tf.transpose(tf.cast(qh, dtype="f32"), perm=(0, 2, 1, 3))
            scaled = queries * tf.full_like(queries, value=SCALE)
            template = tf.reduce(queries, axes=(-1,), keepdim=True, kind="sum")
            m = tf.full_like(template, value=-1e30)
            l = tf.full_like(template, value=0.0)
            acc = tf.full_like(queries, value=0.0)

            for start in tile(CTX, BLOCK):
                base = start + 0
                kb = tf.reshard(
                    tf.repeat_interleave(
                        k_cache[:, base : base + BLOCK, :, :],
                        repeats=GQA_GROUP,
                        axis=2,
                    ),
                    (1, BLOCK, QUERY_HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                vb = tf.reshard(
                    tf.repeat_interleave(
                        v_cache[:, base : base + BLOCK, :, :],
                        repeats=GQA_GROUP,
                        axis=2,
                    ),
                    (1, BLOCK, QUERY_HEADS @ cta.head, HEAD_DIM),
                    "smem",
                )
                keys = tf.transpose(tf.cast(kb, dtype="f32"), perm=(0, 2, 3, 1))
                values = tf.transpose(tf.cast(vb, dtype="f32"), perm=(0, 2, 1, 3))
                scores = tf.matmul(scaled, keys)
                block_m = tf.reduce(scores, axes=(-1,), keepdim=True, kind="max")
                next_m = tf.max(m, block_m)
                correction = tf.exp(m - next_m)
                weights = tf.exp(scores - next_m)
                l = l * correction + tf.reduce(
                    weights, axes=(-1,), keepdim=True, kind="sum"
                )
                acc = acc * correction + tf.matmul(weights, values)
                m = next_m

            k_current = tf.repeat_interleave(
                tf.index_select(k_all, cur_pos, dim=1), repeats=GQA_GROUP, axis=2
            )
            v_current = tf.repeat_interleave(
                tf.index_select(v_all, cur_pos, dim=1), repeats=GQA_GROUP, axis=2
            )
            k_current = tf.reshard(
                k_current, (1, 1, QUERY_HEADS @ cta.head, HEAD_DIM), "smem"
            )
            v_current = tf.reshard(
                v_current, (1, 1, QUERY_HEADS @ cta.head, HEAD_DIM), "smem"
            )
            current_keys = tf.transpose(
                tf.cast(k_current, dtype="f32"), perm=(0, 2, 3, 1)
            )
            current_values = tf.transpose(
                tf.cast(v_current, dtype="f32"), perm=(0, 2, 1, 3)
            )
            current_scores = tf.matmul(scaled, current_keys)
            current_m = tf.max(m, current_scores)
            current_correction = tf.exp(m - current_m)
            current_weights = tf.exp(current_scores - current_m)
            l = l * current_correction + current_weights
            acc = acc * current_correction + tf.matmul(
                current_weights, current_values
            )

            attended = tf.transpose(
                tf.cast(acc / l, dtype="bf16"), perm=(0, 2, 1, 3)
            )
            attended = tf.reshard(
                attended, (1, 1, QUERY_HEADS, HEAD_DIM), "gmem"
            )
            return tf.matmul(tf.reshape(attended, new_shape=(1, 1, HIDDEN)), w_o)


gqa_decode_cache_prepared = Stage5_CachePrepared.entry_function()


__all__ = [
    "BLOCK",
    "CTX",
    "GQA_GROUP",
    "HEAD_DIM",
    "HIDDEN",
    "KV_DIM",
    "KV_HEADS",
    "QUERY_HEADS",
    "ROPE_CONTEXT",
    "SMEM_BUDGET",
    "SPECIALIZE_T",
    "Stage0_Naive",
    "Stage1_Specialized",
    "Stage2_Sharded",
    "Stage3_Fused",
    "Stage4_WeightPrepared",
    "Stage5_CachePrepared",
    "WORKERS",
    "gqa_decode",
    "gqa_decode_specialized",
    "gqa_decode_sharded",
    "gqa_decode_fused",
    "gqa_decode_weight_prepared",
    "gqa_decode_cache_prepared",
    "main",
    "render_markdown",
]


_ANALYSIS = ("compute-cost", "memory", "roofline")
_STAGE_NAMES = (
    "Stage0_Naive",
    "Stage1_Specialized",
    "Stage2_Sharded",
    "Stage3_Fused",
    "Stage4_WeightPrepared",
    "Stage5_CachePrepared",
)
_STAGES = {
    name: globals()[name]
    for name in _STAGE_NAMES
}
_REPORTS: dict[tuple[str, int, bool], tuple[str, str]] = {}


def _analyze_stage(
    name: str, ctx_len: int, *, operands: bool = False
) -> tuple[str, str]:
    """Run the same analysis API used by the CLI and retain its renderings."""
    key = (name, ctx_len, operands)
    if key in _REPORTS:
        return _REPORTS[key]
    stage = _STAGES[name]
    result = run_analysis(
        stage,
        stage.entry_function(),
        analysis=_ANALYSIS,
        dims={"ctx_len": ctx_len},
    )
    rendering = render_analysis(result, operands=operands)
    value = (render_text(rendering), rendering.annotated)
    _REPORTS[key] = value
    return value


def _summary_line(report: str, prefix: str) -> str:
    """Select one stable summary line from a rendered report."""
    for line in report.splitlines():
        if line.startswith(prefix):
            return line
    raise RuntimeError(f"report has no line starting with {prefix!r}")


def _metric(report: str, pattern: str) -> str:
    match = re.search(pattern, report)
    if match is None:
        raise RuntimeError(f"report has no metric matching {pattern!r}")
    return match.group(1)


def _fenced(text: str, language: str = "text") -> str:
    return f"```{language}\n{text.rstrip()}\n```"


def _command(
    stage: str,
    report_name: str,
    ctx_len: int,
    *,
    operands: bool = False,
    as_json: bool = False,
) -> str:
    flags = ["--compute-cost", "--memory", "--roofline"]
    if operands:
        flags.append("--operands")
    flags.append(f"--dim ctx_len={ctx_len}")
    if as_json:
        flags.append("--json")
    return "\n".join(
        (
            f"tilefoundry analyze tests/fixtures/tutorial/attn_layer.py:{stage} \\",
            f"  /tmp/tilefoundry-tutorial-gqa/{report_name} \\",
            f"  {' '.join(flags)}",
        )
    )


def _annotated_line(annotated: str, needle: str) -> str:
    for line in annotated.splitlines():
        if needle in line:
            return line.rstrip()
    raise RuntimeError(f"annotated report has no line containing {needle!r}")


def _annotated_block(annotated: str, needle: str) -> str:
    """Keep one multi-line reshard call without copying it into the page."""
    lines = annotated.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if needle in line),
        None,
    )
    if start is None:
        raise RuntimeError(f"annotated report has no block containing {needle!r}")
    end = start
    while end + 1 < len(lines):
        end += 1
        if end > start and "  # " in lines[end]:
            break
    return "\n".join(line.rstrip() for line in lines[start : end + 1])


def _report_with_annotations(
    name: str, ctx_len: int, needles: tuple[str, ...]
) -> str:
    report, annotated = _analyze_stage(name, ctx_len, operands=True)
    selected: list[str] = []
    for needle in needles:
        if needle in {"reshard(w_q", "reshard(w_o"}:
            selected.append(_annotated_block(annotated, needle))
        else:
            selected.append(_annotated_line(annotated, needle))
    body = report if not selected else f"{report}\n\n" + "\n".join(selected)
    return _fenced(body)


def _capacity_refusal(name: str, ctx_len: int) -> str:
    try:
        _analyze_stage(name, ctx_len)
    except Exception as error:
        return f"tilefoundry: error: {error}"
    raise RuntimeError(f"{name} unexpectedly accepted ctx_len={ctx_len}")


def _stage_link(name: str) -> str:
    return f"[{name}](../../tests/fixtures/tutorial/attn_layer.py)"


def _summary_metrics(report: str) -> tuple[str, str, str, str, str]:
    compute = _summary_line(report, "# compute-cost ")
    traffic = _summary_line(report, "# traffic ")
    peak = _summary_line(report, "# peak-footprint=")
    roofline = _summary_line(report, "# roofline ")
    f32 = _metric(compute, r"f32:([^ ]+)")
    traffic_value = traffic.removeprefix("# traffic traffic=")
    gmem_peak = _metric(peak, r"gmem:([^,]+)")
    ideal, bound = re.search(
        r"ideal-ns=([^ ]+) bound-by=([^ ]+)", roofline
    ).groups()
    return f32, traffic_value, gmem_peak, ideal, bound


def render_markdown() -> str:
    """Execute the tutorial evidence cells and return the complete Markdown page."""
    sweep_contexts = (128, 512, 1024, 2048, 4096, 8192)
    sweep_rows: list[str] = []
    sweep_reports: dict[int, str] = {}
    for ctx_len in sweep_contexts:
        report, _annotated = _analyze_stage("Stage0_Naive", ctx_len, operands=True)
        sweep_reports[ctx_len] = report
        f32, traffic, peak, ideal, bound = _summary_metrics(report)
        sweep_rows.append(f"| {ctx_len} | `{f32}` | `{traffic}` | {peak} | {ideal} | {bound} |")

    stage0_report = sweep_reports[128]
    _stage0_report, stage0_annotated = _analyze_stage("Stage0_Naive", 128, operands=True)
    stage2_short_report, _stage2_short_annotated = _analyze_stage(
        "Stage2_Sharded", 128, operands=True
    )
    stage2_boundary_report, _stage2_boundary_annotated = (
        _analyze_stage("Stage2_Sharded", SPECIALIZE_T, operands=True)
    )
    refusal = _capacity_refusal("Stage2_Sharded", SPECIALIZE_T + 4)
    stage3_report = _report_with_annotations("Stage3_Fused", 4096, ("cache_update(k_cache",))
    stage4_report = _report_with_annotations(
        "Stage4_WeightPrepared", 4096, ("reshard(w_q", "reshard(w_o")
    )
    stage5_report = _report_with_annotations(
        "Stage5_CachePrepared", 4096, ("slice(k_cache", "cache_update(k_cache")
    )

    stage3_f32, stage3_traffic, stage3_peak, stage3_ideal, _stage3_bound = (
        _summary_metrics(stage3_report.removesuffix("\n```").removeprefix("```text\n"))
    )
    stage4_f32, stage4_traffic, stage4_peak, stage4_ideal, _stage4_bound = (
        _summary_metrics(stage4_report.removesuffix("\n```").removeprefix("```text\n"))
    )
    stage5_f32, stage5_traffic, stage5_peak, stage5_ideal, _stage5_bound = (
        _summary_metrics(stage5_report.removesuffix("\n```").removeprefix("```text\n"))
    )
    stage2_boundary_summary = stage2_boundary_report

    report_body = [
        "# Authoring a decode GQA kernel",
        "",
        "This page is generated by executing `tests/fixtures/tutorial/attn_layer.py`.",
        "The six Modules and this page therefore have one source file: change the source,",
        "run it, and the Markdown report is rebuilt from the current analysis results.",
        "",
        _fenced(
            "write a complete shape\n"
            "        |\n"
            "        v\n"
            "analyze one size and sweep ctx_len\n"
            "        |\n"
            "        v\n"
            "change one placement decision\n"
            "        |\n"
            "        v\n"
            "analyze again -> keep the evidence -> choose the next decision",
            "text",
        ),
        "",
        "The six programs are in one executable file:",
        "",
        "| stage | source | decision |",
        "|---|---|---|",
        f"| 0 | {_stage_link('Stage0_Naive')} | one unsplit CTA, open `ctx_len` |",
        f"| 1 | {_stage_link('Stage1_Specialized')} | dispatch on a measured range |",
        f"| 2 | {_stage_link('Stage2_Sharded')} | split query heads across CTAs |",
        f"| 3 | {_stage_link('Stage3_Fused')} | split the KV scan and combine online state |",
        f"| 4 | {_stage_link('Stage4_WeightPrepared')} | stage projection weights by output slice |",
        f"| 5 | {_stage_link('Stage5_CachePrepared')} | stream cache blocks through smem |",
        "",
        "All reports below are produced by the same `analyze()` API used by the CLI.",
        "The reports are static analysis results; no CUDA kernel is launched.",
        "",
        "## 0. Start with the complete shape",
        "",
        "The small teaching shape keeps the published GQA ratio:",
        "",
        "| value | extent |",
        "|---|---:|",
        f"| hidden | {HIDDEN} |",
        f"| query heads | {QUERY_HEADS} |",
        f"| KV heads | {KV_HEADS} |",
        f"| head dimension | {HEAD_DIM} |",
        f"| prior cache | `ctx_len` in `[1, {ROPE_CONTEXT + 1})` |",
        "",
        "The entry includes q/k/v projection, RoPE, one cache append, an attention scan,",
        "and the output projection. The weights are `ConstTensor` parameters. The starting",
        "point has no explicit mesh and no storage transition.",
        "",
        "Run one point like this:",
        "",
        _fenced(_command("Stage0_Naive", "stage0-128.txt", 128, operands=True), "bash"),
        "",
        "The report header is:",
        "",
        _fenced(stage0_report),
        "",
        "The first number before `@` is global work or traffic. The number after `@` is the",
        "per-CTA projection. With no authored split they are equal. The report also annotates",
        "direct calls. These lines are from the same generated report:",
        "",
        _fenced(
            "\n".join(
                (
                    _annotated_line(stage0_annotated, "matmul(hidden, w_q)"),
                    _annotated_line(stage0_annotated, "cache_update(k_cache"),
                    _annotated_line(stage0_annotated, "matmul(v33, w_o)"),
                )
            )
        ),
        "",
        f"`w_q` and `w_o` are each `{HIDDEN} * {HIDDEN} * 2 = {HIDDEN * HIDDEN * 2}` bytes. "
        f"`w_k` and `w_v` are each `{HIDDEN} * {KV_DIM} * 2 = {HIDDEN * KV_DIM * 2}` bytes, "
        f"so the four projection weights total `{2 * HIDDEN * HIDDEN * 2 + 2 * HIDDEN * KV_DIM * 2}` bytes. "
        "That fixed amount is separate from the cache scan.",
        "",
        "## 1. Sweep the open dimension",
        "",
        "The same command, with a different report path, produces the table:",
        "",
        _fenced(
            "mkdir -p /tmp/tilefoundry-tutorial-gqa\n"
            "for ctx in 128 512 1024 2048 4096 8192; do\n"
            "  tilefoundry analyze tests/fixtures/tutorial/attn_layer.py:Stage0_Naive \\\n"
            "    /tmp/tilefoundry-tutorial-gqa/stage0-$ctx.txt \\\n"
            "    --compute-cost --memory --roofline --operands --dim ctx_len=$ctx\n"
            "done",
            "bash",
        ),
        "",
        "The report fields are:",
        "",
        "| `ctx_len` | f32 flops `global@CTA` | traffic `global@CTA` | peak gmem bytes | ideal ns | bound |",
        "|---:|---:|---|---:|---:|---|",
        *sweep_rows,
        "",
        "The table says:",
        "",
        _fenced(
            "per-CTA work = global work        -> no authored split yet\n"
            "weight bytes = fixed              -> projection weights are a staging target\n"
            "cache scan   = grows with ctx_len -> a full-cache residency decision will fail first"
        ),
        "",
        "## 2. Specialize at the capacity boundary",
        "",
        f"The full-cache sharded program in {_stage_link('Stage2_Sharded')} places one local "
        "query head's K and V cache in smem. The boundary is derived from the target capacity:",
        "",
        _fenced(
            f"bytes per ctx per CTA = K + V\n"
            f"                       = 2 * HEAD_DIM * sizeof(bf16)\n"
            f"                       = 2 * {HEAD_DIM} * 2\n"
            f"                       = {CACHE_BYTES_PER_CONTEXT_PER_CTA} B\n\n"
            f"T = floor({SMEM_BUDGET} B / {CACHE_BYTES_PER_CONTEXT_PER_CTA} B)\n"
            f"  = {SPECIALIZE_T}"
        ),
        "",
        f"{_stage_link('Stage1_Specialized')} expresses the dispatch as two half-open "
        f"`DimVarRangePat` variants: `[1, {SPECIALIZE_T})` and `[{SPECIALIZE_T}, {ROPE_CONTEXT + 1})`. "
        "The Stage1 body is deliberately the unsplit baseline, so the dispatch contract can be "
        "read independently from the later implementations.",
        "",
        "The valid boundary report starts with:",
        "",
        _fenced(_command("Stage2_Sharded", "stage2-1816.txt", SPECIALIZE_T), "bash"),
        "",
        _fenced(stage2_boundary_summary),
        "",
        f"A larger context crosses the stated capacity. The generator captured this refusal at "
        f"`ctx_len={SPECIALIZE_T + 4}`:",
        "",
        _fenced(
            _command("Stage2_Sharded", f"stage2-{SPECIALIZE_T + 4}.txt", SPECIALIZE_T + 4),
            "bash",
        ),
        "",
        _fenced(refusal),
        "",
        "The formula chooses the dispatch boundary. It is not a benchmark-tuned magic number.",
        "",
        "## 3. Split the query heads",
        "",
        f"The next change is {_stage_link('Stage2_Sharded')}: one `cta.head` owns one query head. "
        "The same-size comparison isolates placement from context growth.",
        "",
        "Baseline at `ctx_len=128`:",
        "",
        _fenced(_command("Stage0_Naive", "stage0-128.txt", 128), "bash"),
        "",
        _fenced("\n".join(_summary_line(stage0_report, prefix) for prefix in (
            "# compute-cost ", "# traffic ", "# peak-footprint=", "# roofline "
        ))),
        "",
        "Head-sharded at `ctx_len=128`:",
        "",
        _fenced(_command("Stage2_Sharded", "stage2-128.txt", 128), "bash"),
        "",
        _fenced("\n".join(_summary_line(stage2_short_report, prefix) for prefix in (
            "# compute-cost ", "# traffic ", "# peak-footprint=", "# roofline "
        ))),
        "",
        "The f32 work and special work divide by the eight head CTAs. The small bf16 difference is",
        "placement and gather overhead that is not head-shardable. `ideal-ns` also changes because",
        "the authored storage choices change the traffic seen by the roofline calculation.",
        "",
        "## 4. Keep the scan state on chip",
        "",
        f"At long context, the full-cache form is the wrong residency choice. {_stage_link('Stage3_Fused')} "
        "uses a two-dimensional CTA mesh. The head axis owns query heads and the worker axis owns "
        "disjoint cache blocks. Each worker keeps online `(m, l, acc)` state, then the worker axis "
        "is combined with an explicit log-sum-exp merge.",
        "",
        _fenced(_command("Stage3_Fused", "stage3-4096.txt", 4096, operands=True), "bash"),
        "",
        stage3_report,
        "",
        f"At `ctx_len=4096`, gmem read is `{stage3_traffic}` and the report's ideal bound is "
        f"`{stage3_ideal} ns`; the f32 per-CTA work is `{stage3_f32}` and the gmem peak is "
        f"`{stage3_peak}` bytes. These are authored-model bounds, not measured kernel times.",
        "The compact [flash split-K fixture](../../tests/fixtures/placed/flash_split_k_decode.py) "
        "is the matching minimal example.",
        "",
        "This program uses explicit state and does not create an authored `Partial` value. A split-K",
        "algorithm and a `Partial` shard attribute are related ideas, not interchangeable syntax.",
        "",
        "## 5. Stage projection weights",
        "",
        f"{_stage_link('Stage4_WeightPrepared')} moves each projection weight's output slice to smem "
        "before the matmul. The q and o weight lines show the per-CTA read:",
        "",
        stage4_report,
        "",
        _fenced(_command("Stage4_WeightPrepared", "stage4-4096.txt", 4096, operands=True), "bash"),
        "",
        f"The generated header reports f32 `{stage4_f32}`, traffic `{stage4_traffic}`, peak gmem "
        f"`{stage4_peak}` bytes, and ideal bound `{stage4_ideal} ns`. This is kernel staging. "
        "Runtime `Module.load` and a weight converter are a different contract. The real converter "
        "example is [Gemma's `lm_head.converter`](../../tests/models/gemma2_2b/model.py), and the "
        "workflow explanation is in [migrate](migrate.md).",
        "",
        "## 6. Stream the KV cache",
        "",
        f"{_stage_link('Stage5_CachePrepared')} leaves the static projection weights in their ordinary "
        "form and changes only the cache scan. `BLOCK=128` rows move through smem while `(m, l, acc)` "
        "stays resident. The updated cache is read at `cur_pos` once, so append and scan are separate "
        "traffic events.",
        "",
        _fenced(_command("Stage5_CachePrepared", "stage5-4096.txt", 4096, operands=True), "bash"),
        "",
        stage5_report,
        "",
        f"The generated report gives f32 `{stage5_f32}`, traffic `{stage5_traffic}`, peak gmem "
        f"`{stage5_peak}` bytes, and ideal bound `{stage5_ideal} ns`. The distinction is:",
        "",
        _fenced(
            "weight staging: static tensor -> one output slice -> reusable for the step\n"
            "cache staging:  growing context -> one block -> compute -> next block"
        ),
        "",
        "## Feature ledger",
        "",
        "The page uses the new ladder plus existing real programs for features orthogonal to GQA:",
        "",
        "| feature | live program |",
        "|---|---|",
        f"| `@module(entry/target/topologies)` | {_stage_link('Stage0_Naive')} |",
        f"| `@func`, `Tensor`, `ConstTensor`, `DimVar` | {_stage_link('Stage0_Naive')} |",
        f"| `pass` prototype and `@f.specialize(DimVarRangePat)` | {_stage_link('Stage1_Specialized')} |",
        f"| single `Mesh`, shard sugar `X @ m.axis`, `reshard` to smem/gmem | {_stage_link('Stage2_Sharded')} |",
        f"| split-K worker mesh and online softmax state | {_stage_link('Stage3_Fused')} |",
        f"| weight staging and output gather | {_stage_link('Stage4_WeightPrepared')} |",
        f"| cache update, block scan, `matmul`, `rope`, `reduce`, `cast` | {_stage_link('Stage5_CachePrepared')} |",
        "| nested Mesh, `rmem`, rank-changing `reshard`, tuple output, multi-output writeback, multi-level `Topology` | [`rmsnorm_quant_seq2.py`](../../tests/fixtures/placed/rmsnorm_quant_seq2.py) |",
        "| two-dimensional split-K placement with explicit combine | [`flash_split_k_decode.py`](../../tests/fixtures/placed/flash_split_k_decode.py) |",
        "| runtime weight converter | [`gemma2_2b/model.py`](../../tests/models/gemma2_2b/model.py) |",
        "",
        "The command surface used by this page is:",
        "",
        _fenced(
            "--compute-cost  logical work and traffic\n"
            "--memory        residency and peak footprint\n"
            "--roofline      ideal bound and limiting resource\n"
            "--performance   per-level execution projection\n"
            "--operands      operand split in annotated call lines\n"
            "--dim           bind ctx_len for one static analysis run\n"
            "--json          write the same report data as JSON"
        ),
        "",
        "For example, the JSON form writes to a path just like the text form:",
        "",
        _fenced(
            _command(
                "Stage0_Naive", "stage0-128.json", 128, as_json=True
            ),
            "bash",
        ),
        "",
        "Two items are intentionally limitations rather than invented coverage:",
        "",
        "- The split-K tutorial programs use explicit log-sum-exp state. There is no authored `Partial` value in this ladder.",
        "- `analyze` reports authored static bounds. It does not replace `check`, a GPU run, cache invalidation, or a benchmark protocol.",
        "",
        "The source-level rules still apply: an `@func` body uses variants instead of Python `if`, a Module",
        "entry is called through its bare binding inside that Module, and a cast that looks redundant may be",
        "the dtype boundary required by `check`. The deliberate bf16 to f32 round trip is visible in",
        "[`rmsnorm_quant_seq2.py`](../../tests/fixtures/placed/rmsnorm_quant_seq2.py).",
    ]
    return "\n".join(report_body).rstrip() + "\n"


def main() -> int:
    """Print the generated authoring page."""
    sys.stdout.write(render_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
