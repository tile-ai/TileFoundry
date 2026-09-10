"""SM90 GMMA XQA paged multi-head attention.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
csrc/xqa/mha_sm90.cu
license: Apache-2.0 (no upstream source is vendored)

The specialization keeps the upstream twelve-warp CTA, GQA head grouping,
split context, and online-softmax output contract.

The blockwise online-softmax state follows
tests/fixtures/placed/flash_split_k_decode.py.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, Q_HEADS, KV_HEADS, HEAD_DIM, CONTEXT = 4, 48, 4, 128, 3072
BLOCK = 64
CTA_COUNT, THREAD_COUNT = BATCH * KV_HEADS, 12 * 32
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class XQAMHASM90:
    """FlashInfer kernel_mha entry for the SM90 path.

    predicted-ns: 66567
    waves: 1
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def run(
        q: Tensor[(BATCH, 1, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        v: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 12, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q,
                (
                    BATCH @ mesh.batch,
                    1,
                    Q_HEADS @ (mesh.kv_head, mesh.warp),
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            q_smem = tf.reshard(
                q_reg,
                (
                    BATCH @ mesh.batch,
                    1,
                    Q_HEADS @ (mesh.kv_head, mesh.warp),
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            query = tf.reshape(
                tf.cast(q_smem, "f32"),
                new_shape=(
                    BATCH,
                    1,
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
                    1,
                    KV_HEADS @ mesh.kv_head,
                    (Q_HEADS // KV_HEADS) @ mesh.warp,
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
                        BLOCK,
                        KV_HEADS @ mesh.kv_head,
                        HEAD_DIM @ mesh.lane,
                    ),
                    "smem",
                )
                v_smem = tf.reshard(
                    v[:, start : start + BLOCK, :, :],
                    (
                        BATCH @ mesh.batch,
                        BLOCK,
                        KV_HEADS @ mesh.kv_head,
                        HEAD_DIM @ mesh.lane,
                    ),
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
                        1,
                        KV_HEADS @ mesh.kv_head,
                        (Q_HEADS // KV_HEADS) @ mesh.warp,
                        1,
                        BLOCK,
                    ),
                    "smem",
                )
                score = score * 0.08838834764831845
                block_max = tf.reduce(score, axes=(-1,), keepdim=True, kind="max")
                next_max = tf.max(running_max, block_max)
                correction = tf.exp(running_max - next_max)
                weight = tf.exp(score - next_max)
                next_sum = running_sum * correction + tf.reduce(
                    weight, axes=(-1,), keepdim=True, kind="sum"
                )
                block_out = tf.matmul(weight, value)
                next_out = running_out * correction + block_out
                running_max = next_max
                running_sum = next_sum
                running_out = next_out

            output = tf.reshape(
                tf.cast(running_out / running_sum, "bf16"),
                new_shape=(BATCH, 1, Q_HEADS, HEAD_DIM),
            )
            return tf.reshard(
                output,
                (
                    BATCH @ mesh.batch,
                    1,
                    Q_HEADS @ (mesh.kv_head, mesh.warp),
                    HEAD_DIM @ mesh.lane,
                ),
                "gmem",
            )


__all__ = ["XQAMHASM90"]
