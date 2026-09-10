"""Placed band-mask construction used by FlashInfer CuTe attention.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/fusion/mask.py
license: Apache-2.0 (no upstream source is vendored)

CTA query rows and thread-partitioned KV columns retain causal, left-window,
right-window, and sequence-tail bounds before materializing the mask.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 2, 128, 2048
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", BATCH * QUERY), Topology("thread", 4 * 32))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AttentionMaskModule:
    """FlashInfer AttentionMask entry.

    predicted-ns: 1952  waves: 2
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def run(scores: Tensor[(BATCH, QUERY, CONTEXT), "f32"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, QUERY, 4, 32),
            names=("batch", "query", "warp", "lane"),
        ) as mesh:
            score_reg = tf.reshard(
                scores,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "rmem",
            )
            q_pos = tf.reshape(
                tf.arange(Tensor[(QUERY,), "i64"]),
                new_shape=(1, QUERY, 1),
            )
            k_pos = tf.reshape(
                tf.arange(Tensor[(CONTEXT,), "i64"]),
                new_shape=(1, 1, CONTEXT),
            )
            q_reg = tf.reshard(
                q_pos,
                (1, QUERY @ mesh.query, 1),
                "rmem",
            )
            k_reg = tf.reshard(
                k_pos,
                (1, 1, 8 @ mesh.warp, 256 @ mesh.lane),
                "rmem",
            )
            offset = CONTEXT - QUERY
            visible = (k_reg <= q_reg + offset) and (k_reg >= q_reg + offset - 512)
            masked = tf.where(visible, score_reg, -1000000.0)
            mask_smem = tf.reshard(
                masked,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "smem",
            )
            return tf.reshard(
                mask_smem,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


__all__ = ["AttentionMaskModule"]
