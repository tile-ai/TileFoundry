"""A tensor resharded from rows to columns across two cards.

Splitting the same data two ways is an all-to-all: every card keeps the part
both shards give it and sends the rest. Nothing here executes that exchange --
the program exists so an analysis can say what it would cost, which is what
``tests/analysis/test_analysis_families.py`` asks of it: that the crossing is
``SENT_BYTES`` for one card and that a unit inside the boundary states no
share of it.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

GPUS, CTAS, R, C = 2, 4, 8, 8
GPU, CTA = Topology("gpu", GPUS), Topology("cta", CTAS)


HELD_BYTES = R * C * 4 // GPUS
SENT_BYTES = HELD_BYTES - HELD_BYTES // GPUS


@module(
    entry="transpose_shard",
    target=CudaTarget("nvidia.h200_sxm", device_count=GPUS),
    topologies=(GPU, CTA),
)
class TransposeShard:
    """One card's rows become one card's columns, so most of its share leaves."""

    @func
    def transpose_shard(
        x: Tensor[(R, C), "f32", ShardLayout(Layout((GPUS, R // GPUS, C), (R // GPUS * C, C, 1)), (Split(0),), Mesh((GPU,), Layout((GPUS,), (1,)), ("g",)))],
    ) -> Tensor[(R, C), "f32", ShardLayout(Layout((R, GPUS, C // GPUS), (C, C // GPUS, 1)), (Split(1),), Mesh((GPU,), Layout((GPUS,), (1,)), ("g",)))]:
        with Mesh(("gpu",), layout=(GPUS,), names=("g",)) as g:
            return tf.reshard(x, (R, C @ g.g), "gmem")
