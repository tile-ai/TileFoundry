"""SM120 clustered XQA multi-latent attention.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
csrc/xqa/mla_sm120.cu
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: pending primitive rewrite probe
ledger: None

The specialization keeps four CTAs per input token, twelve warps per CTA,
compressed 576-wide KV latent state, and the 128-head MLA output.

The explicit score and online state follow
tests/fixtures/placed/flash_split_k_decode.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, TOKENS, HEADS, QK_DIM, KV_DIM, CONTEXT = 4, 1, 128, 192, 576, 1920
BLOCK = 24
CTA_COUNT, THREAD_COUNT = 4 * BATCH * TOKENS, 12 * 32
TARGET = CudaTarget("nvidia.b200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class XQAMLASM120:
    """FlashInfer kernel_mha entry for the SM120 MLA path."""

    @func
    def run(
        q: Tensor[(BATCH, TOKENS, HEADS, QK_DIM), "bf16"],
        kv: Tensor[(BATCH, CONTEXT, KV_DIM), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, 4, 12, 32),
            names=("batch", "head_group", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (
                    BATCH @ mesh.batch,
                    TOKENS,
                    HEADS @ mesh.head_group,
                    QK_DIM @ mesh.lane,
                ),
                "rmem",
            )
            q_smem = tf.reshard(
                q_reg,
                (
                    BATCH @ mesh.batch,
                    TOKENS,
                    HEADS @ mesh.head_group,
                    QK_DIM @ mesh.lane,
                ),
                "smem",
            )
            query = tf.reshape(
                tf.cast(q_smem, "f32"),
                new_shape=(BATCH, TOKENS, HEADS, 1, QK_DIM),
            )
            state_partial = tf.reduce(query, axes=(-1,), keepdim=True, kind="sum")
            state = tf.reshard(
                state_partial,
                (
                    BATCH @ mesh.batch,
                    TOKENS,
                    HEADS @ mesh.head_group,
                    1,
                    1,
                ),
                "smem",
            )
            running_max = tf.full_like(state, value=-1e30)
            running_sum = tf.full_like(state, value=0.0)
            running_out = tf.full_like(query, value=0.0)

            for start in range(0, CONTEXT, BLOCK):
                latent = kv[:, start : start + BLOCK, :]
                key_smem = tf.reshard(
                    latent[:, :, :QK_DIM],
                    (
                        BATCH @ mesh.batch,
                        BLOCK @ mesh.warp,
                        QK_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                value_smem = tf.reshard(
                    latent[:, :, QK_DIM : 2 * QK_DIM],
                    (
                        BATCH @ mesh.batch,
                        BLOCK @ mesh.warp,
                        QK_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                key = tf.reshape(
                    tf.cast(key_smem, "f32"),
                    new_shape=(BATCH, 1, 1, BLOCK, QK_DIM),
                )
                value = tf.reshape(
                    tf.cast(value_smem, "f32"),
                    new_shape=(BATCH, 1, 1, BLOCK, QK_DIM),
                )
                score_partial = tf.reduce(
                    query * key, axes=(-1,), keepdim=True, kind="sum"
                )
                score = tf.reshard(
                    score_partial,
                    (
                        BATCH @ mesh.batch,
                        TOKENS,
                        HEADS @ mesh.head_group,
                        BLOCK @ mesh.warp,
                        1,
                    ),
                    "smem",
                )
                score = score * 0.07216878364870322
                block_max = tf.reduce(score, axes=(-2,), keepdim=True, kind="max")
                next_max = tf.max(running_max, block_max)
                correction = tf.exp(running_max - next_max)
                weight = tf.exp(score - next_max)
                next_sum = running_sum * correction + tf.reduce(
                    weight, axes=(-2,), keepdim=True, kind="sum"
                )
                block_out_partial = tf.reduce(
                    weight * value, axes=(-2,), keepdim=False, kind="sum"
                )
                block_out = tf.reshard(
                    block_out_partial,
                    (
                        BATCH @ mesh.batch,
                        TOKENS,
                        HEADS @ mesh.head_group,
                        QK_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                next_out = running_out * correction + tf.reshape(
                    block_out,
                    new_shape=(BATCH, TOKENS, HEADS, 1, QK_DIM),
                )
                running_max = next_max
                running_sum = next_sum
                running_out = next_out

            return tf.reshard(
                tf.cast(
                    tf.reshape(
                        running_out / running_sum,
                        new_shape=(BATCH, TOKENS, HEADS, QK_DIM),
                    ),
                    "bf16",
                ),
                (
                    BATCH @ mesh.batch,
                    TOKENS,
                    HEADS @ mesh.head_group,
                    QK_DIM @ mesh.lane,
                ),
                "gmem",
            )


__all__ = ["XQAMLASM120"]
