"""Placed indexed attention composed from HIR primitives.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 include/flashinfer/attention/decode.cuh
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: pending primitive rewrite probe
ledger: None

Page/block gather, score normalization, and value reduction follow the
primitive structure in tests/fixtures/placed/derived_prefill.py, tests/fixtures/placed/mha_decode_paged.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 1920
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * KV_HEADS, 4 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))
MLA_TOPOLOGIES = (Topology("cta", BATCH * 4), Topology("thread", 12 * 32))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class SingleDecodeWithKVCacheKernelModule1:
    """FlashInfer SingleDecodeWithKVCacheKernel entry."""

    @func
    def run(
        q: Tensor[(BATCH, 1, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        v: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        page_indices: Tensor[(BATCH, CONTEXT // 16), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (BATCH @ mesh.batch, 1, Q_HEADS @ (mesh.kv_head, mesh.warp), HEAD_DIM @ mesh.lane),
                "rmem",
            )
            k_smem = tf.reshard(
                k,
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            v_smem = tf.reshard(
                v,
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            pages_reg = tf.reshard(page_indices, (BATCH @ mesh.batch, CONTEXT // 16), "rmem")
            flat_pages = tf.reshape(
                pages_reg, new_shape=(BATCH * (CONTEXT // 16),)
            )
            paged_k = tf.reshard(
                tf.reshape(
                    k_smem,
                    new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                ),
                (BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            paged_v = tf.reshard(
                tf.reshape(
                    v_smem,
                    new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                ),
                (BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            selected_k = tf.reshard(
                tf.reshape(
                    tf.index_select(paged_k, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            selected_v = tf.reshard(
                tf.reshape(
                    tf.index_select(paged_v, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            query = tf.reshape(
                tf.cast(q_reg, "f32"),
                new_shape=(BATCH, 1, KV_HEADS, Q_HEADS // KV_HEADS, 1, HEAD_DIM),
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
            weight = tf.exp(score - peak)
            total = tf.reduce(weight, axes=(-2,), keepdim=False, kind="sum")
            blended = tf.reduce(weight * value, axes=(-2,), keepdim=False, kind="sum")
            return tf.cast(
                tf.reshape(blended / total, new_shape=(BATCH, 1, Q_HEADS, HEAD_DIM)),
                "bf16",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BatchDecodeWithPagedKVCacheKernelModule2:
    """FlashInfer BatchDecodeWithPagedKVCacheKernel entry."""

    @func
    def run(
        q: Tensor[(BATCH, 1, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        v: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        page_indices: Tensor[(BATCH, CONTEXT // 16), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (BATCH @ mesh.batch, 1, Q_HEADS @ (mesh.kv_head, mesh.warp), HEAD_DIM @ mesh.lane),
                "rmem",
            )
            k_smem = tf.reshard(
                k,
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            v_smem = tf.reshard(
                v,
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "smem",
            )
            pages_reg = tf.reshard(page_indices, (BATCH @ mesh.batch, CONTEXT // 16), "rmem")
            flat_pages = tf.reshape(
                pages_reg, new_shape=(BATCH * (CONTEXT // 16),)
            )
            paged_k = tf.reshard(
                tf.reshape(
                    k_smem,
                    new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                ),
                (BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            paged_v = tf.reshard(
                tf.reshape(
                    v_smem,
                    new_shape=(BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                ),
                (BATCH * (CONTEXT // 16), 16, KV_HEADS, HEAD_DIM),
                "gmem",
            )
            selected_k = tf.reshard(
                tf.reshape(
                    tf.index_select(paged_k, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            selected_v = tf.reshard(
                tf.reshape(
                    tf.index_select(paged_v, flat_pages, dim=0),
                    new_shape=(BATCH, CONTEXT, KV_HEADS, HEAD_DIM),
                ),
                (BATCH @ mesh.batch, CONTEXT, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            query = tf.reshape(
                tf.cast(q_reg, "f32"),
                new_shape=(BATCH, 1, KV_HEADS, Q_HEADS // KV_HEADS, 1, HEAD_DIM),
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
            weight = tf.exp(score - peak)
            total = tf.reduce(weight, axes=(-2,), keepdim=False, kind="sum")
            blended = tf.reduce(weight * value, axes=(-2,), keepdim=False, kind="sum")
            return tf.cast(
                tf.reshape(blended / total, new_shape=(BATCH, 1, Q_HEADS, HEAD_DIM)),
                "bf16",
            )


@module(entry="run", target=TARGET, topologies=MLA_TOPOLOGIES)
class BatchDecodeWithPagedKVCacheKernelMLAModule3:
    """FlashInfer BatchDecodeWithPagedKVCacheKernelMLA entry."""

    @func
    def run(
        q: Tensor[(BATCH, 1, 128, 192), "bf16"],
        compressed_kv: Tensor[(BATCH, CONTEXT, 576), "bf16"],
        sparse_indices: Tensor[(BATCH, CONTEXT // 16), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, 4, 12, 32),
            names=("batch", "head_group", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, (BATCH @ mesh.batch, 1, 128 @ mesh.head_group, 192 @ mesh.lane), "rmem"
            )
            kv_smem = tf.reshard(
                compressed_kv, (BATCH @ mesh.batch, CONTEXT @ mesh.warp, 576 @ mesh.lane), "smem"
            )
            index_reg = tf.reshard(
                sparse_indices, (BATCH @ mesh.batch, (CONTEXT // 16) @ mesh.warp), "rmem"
            )
            flat_blocks = tf.reshape(
                index_reg, new_shape=(BATCH * (CONTEXT // 16),)
            )
            blocked_kv = tf.reshard(
                tf.reshape(
                    kv_smem, new_shape=(BATCH * (CONTEXT // 16), 16, 576)
                ),
                (BATCH * (CONTEXT // 16), 16, 576),
                "gmem",
            )
            selected_kv = tf.reshape(
                tf.index_select(blocked_kv, flat_blocks, dim=0),
                new_shape=(BATCH, CONTEXT, 576),
            )
            key = tf.reshard(
                tf.slice(
                    selected_kv,
                    (0, 0, 0),
                    sizes=(BATCH, CONTEXT, 192),
                    strides=(1, 1, 1),
                ),
                (BATCH @ mesh.batch, CONTEXT, 192 @ mesh.lane),
                "rmem",
            )
            value = tf.reshard(
                tf.slice(
                    selected_kv,
                    (0, 0, 192),
                    sizes=(BATCH, CONTEXT, 192),
                    strides=(1, 1, 1),
                ),
                (BATCH @ mesh.batch, CONTEXT, 192 @ mesh.lane),
                "rmem",
            )
            query = tf.reshape(
                tf.cast(q_reg, "f32"), new_shape=(BATCH, 1, 128, 1, 192)
            )
            key = tf.reshape(
                tf.cast(key, "f32"), new_shape=(BATCH, 1, 1, CONTEXT, 192)
            )
            value = tf.reshape(
                tf.cast(value, "f32"), new_shape=(BATCH, 1, 1, CONTEXT, 192)
            )
            raw = tf.reduce(query * key, axes=(-1,), keepdim=True, kind="sum")
            score = raw * tf.full_like(raw, value=0.07216878364870322)
            peak = tf.reduce(score, axes=(-2,), keepdim=True, kind="max")
            weight = tf.exp(score - peak)
            total = tf.reduce(weight, axes=(-2,), keepdim=False, kind="sum")
            blended = tf.reduce(weight * value, axes=(-2,), keepdim=False, kind="sum")
            return tf.cast(blended / total, "bf16")
