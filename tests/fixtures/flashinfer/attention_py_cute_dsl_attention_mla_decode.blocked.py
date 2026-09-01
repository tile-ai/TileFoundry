"""Placed indexed attention composed from HIR primitives.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/mla_decode.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: pending primitive rewrite probe
ledger: None

Page/block gather, score normalization, and value reduction follow the
primitive structure in tests/fixtures/placed/derived_prefill.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 1920
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * 4, 12 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BlackwellMultiLatentAttentionForwardModule1:
    """FlashInfer BlackwellMultiLatentAttentionForward entry."""

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
