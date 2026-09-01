"""Placed blockwise attention composed from HIR primitives.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/wrappers/batch_prefill.py
license: Apache-2.0 (no upstream source is vendored)

The blockwise score, mask, online normalization, and value reduction follow
tests/fixtures/placed/prefill_decode_attention.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
BLOCK = 64
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * Q_HEADS, 8 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class BatchPrefillCuteDSLWrapperModule1:
    """FlashInfer BatchPrefillCuteDSLWrapper entry.

    predicted-ns: 68434
    waves: 1
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def run(
        q: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        v: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        mask: Tensor[(BATCH, QUERY, CONTEXT), "bool"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, Q_HEADS // KV_HEADS, 8, 32),
            names=("batch", "kv_head", "group", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (
                    BATCH @ mesh.batch,
                    QUERY,
                    Q_HEADS @ (mesh.kv_head, mesh.group),
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            q_smem = tf.reshard(
                q_reg,
                (
                    BATCH @ mesh.batch,
                    QUERY,
                    Q_HEADS @ (mesh.kv_head, mesh.group),
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            query = tf.reshape(
                tf.cast(q_smem, "f32"),
                new_shape=(
                    BATCH,
                    QUERY,
                    KV_HEADS,
                    Q_HEADS // KV_HEADS,
                    1,
                    HEAD_DIM,
                ),
            )
            state_partial = tf.reduce(query, axes=(-1,), keepdim=True, kind="sum")
            state = tf.reshard(
                state_partial,
                (
                    BATCH @ mesh.batch,
                    QUERY,
                    KV_HEADS @ mesh.kv_head,
                    (Q_HEADS // KV_HEADS) @ mesh.group,
                    1,
                    1,
                ),
                "smem",
            )
            running_max = tf.full_like(state, value=-1e30)
            running_sum = tf.full_like(state, value=0.0)
            running_out = tf.full_like(query, value=0.0)

            for start in range(0, CONTEXT, BLOCK):
                k_smem = tf.reshard(
                    k[:, start : start + BLOCK, :, :],
                    (
                        BATCH @ mesh.batch,
                        BLOCK @ mesh.warp,
                        KV_HEADS @ mesh.kv_head,
                        HEAD_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                v_smem = tf.reshard(
                    v[:, start : start + BLOCK, :, :],
                    (
                        BATCH @ mesh.batch,
                        BLOCK @ mesh.warp,
                        KV_HEADS @ mesh.kv_head,
                        HEAD_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                mask_smem = tf.reshard(
                    mask[:, :, start : start + BLOCK],
                    (BATCH @ mesh.batch, QUERY, BLOCK @ mesh.warp),
                    "smem",
                )
                key = tf.transpose(
                    tf.reshape(
                        tf.cast(k_smem, "f32"),
                        new_shape=(BATCH, 1, BLOCK, KV_HEADS, 1, HEAD_DIM),
                    ),
                    perm=(0, 1, 3, 4, 5, 2),
                )
                value = tf.transpose(
                    tf.reshape(
                        tf.cast(v_smem, "f32"),
                        new_shape=(BATCH, 1, BLOCK, KV_HEADS, 1, HEAD_DIM),
                    ),
                    perm=(0, 1, 3, 4, 2, 5),
                )
                score_partial = tf.matmul(query, key)
                score = tf.reshard(
                    score_partial,
                    (
                        BATCH @ mesh.batch,
                        QUERY,
                        KV_HEADS @ mesh.kv_head,
                        (Q_HEADS // KV_HEADS) @ mesh.group,
                        1,
                        BLOCK @ mesh.warp,
                    ),
                    "smem",
                )
                score = score * 0.08838834764831845
                live = tf.where(
                    tf.reshape(
                        mask_smem,
                        new_shape=(BATCH, QUERY, 1, 1, 1, BLOCK),
                    ),
                    score,
                    tf.full_like(score, value=-1e30),
                )
                block_max = tf.reduce(live, axes=(-1,), keepdim=True, kind="max")
                next_max = tf.max(running_max, block_max)
                correction = tf.exp(running_max - next_max)
                weight = tf.exp(live - next_max)
                next_sum = running_sum * correction + tf.reduce(
                    weight, axes=(-1,), keepdim=True, kind="sum"
                )
                block_out_partial = tf.matmul(weight, value)
                block_out = tf.reshard(
                    block_out_partial,
                    (
                        BATCH @ mesh.batch,
                        QUERY,
                        KV_HEADS @ mesh.kv_head,
                        (Q_HEADS // KV_HEADS) @ mesh.group,
                        1,
                        HEAD_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                next_out = running_out * correction + block_out
                running_max = next_max
                running_sum = next_sum
                running_out = next_out

            output = tf.reshape(
                tf.cast(running_out / running_sum, "bf16"),
                new_shape=(BATCH, QUERY, Q_HEADS, HEAD_DIM),
            )
            return tf.reshard(
                output,
                (
                    BATCH @ mesh.batch,
                    QUERY,
                    Q_HEADS @ (mesh.kv_head, mesh.group),
                    HEAD_DIM @ mesh.lane,
                ),
                "gmem",
            )
