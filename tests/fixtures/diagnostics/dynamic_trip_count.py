"""A loop whose trip count the program computes, used to exercise its refusal."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

CTAS = 132
N = CTAS * 128


@module(
    entry="kernel",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS),),
)
class DynamicTripCount:
    @func
    def kernel(x: Tensor[(N,), "f32"], reps: Tensor[(), "i64"]) -> Tensor[(N,), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as m:
            placed = tf.reshard(x, (N @ m.block,), "gmem")
            acc = placed
            bound = reps + 0
            for _step in range(bound):
                acc = tf.square(acc)
            return tf.reshard(acc, (N @ m.block,), "gmem")
