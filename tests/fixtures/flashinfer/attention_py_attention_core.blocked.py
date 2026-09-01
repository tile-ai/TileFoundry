"""Holistic mixed prefill/decode attention over a paged KV cache.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/attention/_core.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: pending primitive rewrite probe
ledger: None

The specialization preserves CSR page ownership, GQA head sharing, optional
attention sinks, and the output plus log-sum-exp contract.

Page-table gather and score normalization follow
tests/fixtures/placed/mha_decode_paged.py.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, Q_HEADS, KV_HEADS, HEAD_DIM = 4, 32, 8, 128
PAGES, PAGE_SIZE, PAGES_PER_REQUEST = 128, 16, 32
CONTEXT = PAGES_PER_REQUEST * PAGE_SIZE
CTA_COUNT, THREAD_COUNT = BATCH * KV_HEADS, 4 * 32
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BatchAttentionModule:
    """FlashInfer BatchAttention entry."""

    @func
    def run(
        q: Tensor[(BATCH, 1, Q_HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM), "bf16"],
        page_indices: Tensor[(BATCH, PAGES_PER_REQUEST), "i32"],
        sink: Tensor[(Q_HEADS,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("request", "kv_head", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (
                    BATCH @ mesh.request,
                    1,
                    Q_HEADS @ (mesh.kv_head, mesh.warp),
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            k_reg = tf.reshard(
                k_cache,
                (PAGES, PAGE_SIZE, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            v_reg = tf.reshard(
                v_cache,
                (PAGES, PAGE_SIZE, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            k_smem = tf.reshard(
                k_reg,
                (PAGES, PAGE_SIZE, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            v_smem = tf.reshard(
                v_reg,
                (PAGES, PAGE_SIZE, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            pages_reg = tf.reshard(page_indices, (BATCH @ mesh.request, PAGES_PER_REQUEST), "rmem")
            sink_reg = tf.reshard(sink, (Q_HEADS @ (mesh.kv_head, mesh.warp),), "rmem")
            flat_pages = tf.reshape(
                pages_reg, new_shape=(BATCH * PAGES_PER_REQUEST,)
            )
            k_pages = tf.reshard(
                k_smem,
                (PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            v_pages = tf.reshard(
                v_smem,
                (PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            selected_k = tf.reshard(
                tf.reshape(
                    tf.index_select(k_pages, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (
                    BATCH @ mesh.request,
                    CONTEXT,
                    KV_HEADS @ mesh.kv_head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            selected_v = tf.reshard(
                tf.reshape(
                    tf.index_select(v_pages, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (
                    BATCH @ mesh.request,
                    CONTEXT,
                    KV_HEADS @ mesh.kv_head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            query = tf.reshape(
                tf.cast(q_reg, "f32"),
                new_shape=(
                    BATCH,
                    1,
                    KV_HEADS,
                    Q_HEADS // KV_HEADS,
                    1,
                    HEAD_DIM,
                ),
            )
            key = tf.reshape(
                tf.transpose(tf.cast(selected_k, "f32"), perm=(0, 2, 1, 3)),
                new_shape=(BATCH, 1, KV_HEADS, 1, CONTEXT, HEAD_DIM),
            )
            value = tf.reshape(
                tf.transpose(tf.cast(selected_v, "f32"), perm=(0, 2, 1, 3)),
                new_shape=(BATCH, 1, KV_HEADS, 1, CONTEXT, HEAD_DIM),
            )
            raw = tf.reduce(query * key, axes=(-1,), keepdim=True, kind="sum")
            score = raw * tf.full_like(raw, value=0.08838834764831845)
            peak = tf.reduce(score, axes=(-2,), keepdim=True, kind="max")
            sink_score = tf.reshape(
                sink_reg,
                new_shape=(1, 1, KV_HEADS, Q_HEADS // KV_HEADS, 1, 1),
            )
            peak = tf.max(peak, sink_score)
            weight = tf.exp(score - peak)
            sink_weight = tf.exp(sink_score - peak)
            total = tf.reduce(weight, axes=(-2,), keepdim=False, kind="sum") + tf.reduce(
                sink_weight,
                axes=(-2,),
                keepdim=False,
                kind="sum",
            )
            blended = tf.reduce(weight * value, axes=(-2,), keepdim=False, kind="sum")
            output = tf.cast(
                tf.reshape(blended / total, new_shape=(BATCH, 1, Q_HEADS, HEAD_DIM)),
                "bf16",
            )
            lse = tf.reshape(
                tf.log(total)
                + tf.reshape(peak, new_shape=(BATCH, 1, KV_HEADS, Q_HEADS // KV_HEADS, 1)),
                new_shape=(BATCH, 1, Q_HEADS),
            )
            return output, lse


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BatchAttentionWithSinkModule:
    """FlashInfer BatchAttentionWithAttentionSinkWrapper entry."""

    kernel = replace(BatchAttentionModule, name="kernel", target=None)

    @func
    def run(
        q: Tensor[(BATCH, 1, Q_HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM), "bf16"],
        page_indices: Tensor[(BATCH, PAGES_PER_REQUEST), "i32"],
        sink: Tensor[(Q_HEADS,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("request", "kv_head", "warp", "lane"),
        ) as mesh:
            q_local = tf.reshard(
                q,
                (
                    BATCH @ mesh.request,
                    1,
                    Q_HEADS @ (mesh.kv_head, mesh.warp),
                    HEAD_DIM @ mesh.lane,
                ),
                "gmem",
            )
            return kernel(q_local, k_cache, v_cache, page_indices, sink)


__all__ = ["BatchAttentionModule", "BatchAttentionWithSinkModule"]
