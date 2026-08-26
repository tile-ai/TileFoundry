"""Placed positional-bias transforms refused by current local contracts.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/fusion/variant.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: Binary: operands have conflicting storage (rmem, smem); a multi-input op requires its concrete operands to share one residency
ledger: REG-11, EXT-03, FE-07

ALiBi reaches the recorded mixed-residency error first. Selecting RPEAttention
independently reaches the preserved table gather, then exceeds smem capacity:
value v4 needs 262144 B in smem, above the target's 232448 B limit.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 2, 128, 2048
HEADS = 32
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", BATCH * QUERY), Topology("thread", 4 * 32))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ALiBiAttentionModule:
    """FlashInfer ALiBiAttention entry with per-head linear position bias."""

    @func
    def run(
        scores: Tensor[(BATCH, QUERY, CONTEXT), "f32"],
        slopes: ConstTensor[(HEADS,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, QUERY, 4, 32),
            names=("batch", "query", "warp", "lane"),
        ) as mesh:
            scores_reg = tf.reshard(
                scores,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "rmem",
            )
            scores_smem = tf.reshard(
                scores_reg,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "smem",
            )
            positions = tf.reshape(
                tf.arange(Tensor[(CONTEXT,), "i64"]),
                new_shape=(1, 1, CONTEXT),
            )
            positions_reg = tf.reshard(
                positions,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "rmem",
            )
            slope_smem = tf.reshard(slopes, (HEADS @ mesh.lane,), "smem")
            slope = tf.reduce(slope_smem, axes=(0,), keepdim=True, kind="max")
            biased = scores_smem + tf.cast(positions_reg, "f32") * slope
            probabilities = tf.softmax(biased, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class RPEAttentionModule:
    """FlashInfer RPEAttention entry with a learned relative-position table."""

    @func
    def run(
        scores: Tensor[(BATCH, QUERY, CONTEXT), "f32"],
        table: ConstTensor[(HEADS, 129), "f32"],
        relative_indices: Tensor[(CONTEXT,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, QUERY, 4, 32),
            names=("batch", "query", "warp", "lane"),
        ) as mesh:
            scores_reg = tf.reshard(
                scores,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "rmem",
            )
            scores_smem = tf.reshard(
                scores_reg,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "smem",
            )
            table_smem = tf.reshard(table, (HEADS @ mesh.lane, 129), "smem")
            indices_reg = tf.reshard(relative_indices, (CONTEXT @ mesh.warp,), "rmem")
            bias = tf.index_select(table_smem, indices_reg, dim=1)
            biased = scores_smem + tf.reduce(bias, axes=(0,), keepdim=True, kind="max")
            probabilities = tf.softmax(biased, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


__all__ = ["ALiBiAttentionModule", "RPEAttentionModule"]
