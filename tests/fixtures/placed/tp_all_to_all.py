"""A tensor resharded from rows to columns across two cards.

Splitting the same data two ways is an all-to-all: every card keeps the part
both shards give it and sends the rest. Nothing here executes that exchange --
the program exists so an analysis can say what it would cost.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.types.shard import Layout, Split, canonical_shard_layout
from tilefoundry.ir.types.shard import Mesh as ShardMesh
from tilefoundry.target import CudaTarget

GPUS, CTAS, R, C = 2, 4, 8, 8
GPU, CTA = Topology("gpu", GPUS), Topology("cta", CTAS)

_MESH = ShardMesh(topologies=(GPU,), layout=Layout((GPUS,), (1,)), names=("g",))
BY_ROW = canonical_shard_layout((R, C), _MESH, (Split(0),))
BY_COLUMN = canonical_shard_layout((R, C), _MESH, (Split(1),))

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
        x: Tensor[(R, C), "f32", BY_ROW],
    ) -> Tensor[(R, C), "f32", BY_COLUMN]:
        with Mesh(("gpu",), layout=(GPUS,), names=("g",)) as g:
            return tf.reshard(x, (R, C @ g.g), "gmem")
