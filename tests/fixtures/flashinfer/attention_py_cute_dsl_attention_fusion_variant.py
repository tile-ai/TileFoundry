"""Placed score and statistics transforms for runnable CuTe attention variants.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/cute_dsl/attention/fusion/variant.py
license: Apache-2.0 (no upstream source is vendored)

CTA query rows and thread-partitioned context columns preserve each entry's
default, sink, sigmoid, or soft-capping score mechanism.
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
class AttentionVariantModule:
    """FlashInfer AttentionVariant entry with its default softmax behavior.

    predicted-ns: 914  waves: 2
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
            probabilities = tf.softmax(scores_smem, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class StandardAttentionModule:
    """FlashInfer StandardAttention entry with unmodified softmax logits.

    predicted-ns: 914  waves: 2
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
            probabilities = tf.softmax(scores_smem, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AttentionWithSinkModule:
    """FlashInfer AttentionWithSink entry with one learned sink per head.

    predicted-ns: 934  waves: 2
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def run(
        scores: Tensor[(BATCH, QUERY, CONTEXT), "f32"],
        sinks: ConstTensor[(HEADS,), "f32"],
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
            sink_smem = tf.reshard(sinks, (HEADS @ mesh.lane,), "smem")
            sink_stat = tf.reduce(sink_smem, axes=(0,), keepdim=True, kind="max")
            probabilities = tf.softmax(scores_smem - sink_stat, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class SigmoidAttentionModule:
    """FlashInfer SigmoidAttention entry using exp2 and reciprocal logits.

    predicted-ns: 1064  waves: 2
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
            weights = 1.0 / (1.0 + tf.exp2(-scores_smem))
            return tf.reshard(
                weights,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class SigmoidTanhAttentionModule:
    """FlashInfer SigmoidTanhAttention entry using the tanh sigmoid identity.

    predicted-ns: 944  waves: 2
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
            weights = 0.5 + 0.5 * tf.tanh(scores_smem * 0.5)
            return tf.reshard(
                weights,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class SoftCappingAttentionModule:
    """FlashInfer SoftCappingAttention entry with bounded tanh logits.

    predicted-ns: 944  waves: 2
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
            capped = 50.0 * tf.tanh(scores_smem / 50.0)
            probabilities = tf.softmax(capped, axis=-1)
            return tf.reshard(
                probabilities,
                (BATCH @ mesh.batch, QUERY @ mesh.query, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


__all__ = [
    "AttentionVariantModule",
    "AttentionWithSinkModule",
    "SigmoidAttentionModule",
    "SigmoidTanhAttentionModule",
    "SoftCappingAttentionModule",
    "StandardAttentionModule",
]
