"""A row-to-column ownership change across two cards.

Each card's global memory is its own, so a Split over cards is private and the
transpose moves every element a card does not keep to the card that owns it:
the result is a new buffer, and the move is communication.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

GPUS, CTAS, R, C = 2, 4, 8, 8
GPU, CTA = Topology("gpu", GPUS), Topology("cta", CTAS)


HELD_BYTES = R * C * 4 // GPUS
SENT_BYTES = HELD_BYTES - HELD_BYTES // GPUS


@module(
    entry="transpose_shard",
    target=cuda.Target("nvidia.h200_sxm", device_count=GPUS),
    topologies=(GPU, CTA),
)
class TransposeShard:
    """One card's rows become one card's columns, so most of its share leaves."""

    @func
    def transpose_shard(
        x: Tensor[(R, C), "f32", ShardLayout(Layout((GPUS, R // GPUS, C), (0, C, 1)), (Split(0),), Mesh((GPU,), Layout((GPUS,), (1,)), ("g",)))],
    ) -> Tensor[(R, C), "f32", ShardLayout(Layout((R, GPUS, C // GPUS), (C // GPUS, 0, 1)), (Split(1),), Mesh((GPU,), Layout((GPUS,), (1,)), ("g",)))]:
        with Mesh(("gpu",), layout=(GPUS,), names=("g",)) as g:
            return tf.reshard(x, (R, C @ g.g), "gmem")
