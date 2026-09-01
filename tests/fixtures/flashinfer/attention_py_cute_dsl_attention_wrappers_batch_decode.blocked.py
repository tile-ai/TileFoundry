"""Placed indexed attention composed from HIR primitives.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/wrappers/batch_decode.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: pending primitive rewrite probe
ledger: None

Page/block gather, score normalization, and value reduction follow the
primitive structure in tests/fixtures/placed/mha_decode_paged.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * KV_HEADS, 4 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BatchDecodeCuteDSLWrapperModule1:
    """FlashInfer BatchDecodeCuteDSLWrapper entry."""

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
class BatchDecodePagedCuteDSLWrapperModule2:
    """FlashInfer BatchDecodePagedCuteDSLWrapper entry."""

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
