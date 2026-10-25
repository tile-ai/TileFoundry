"""A local view of a thread-sharded tensor, with a static and a symbolic extent."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_N = DimVar("n", 1, 128)


@module(
    entry="main",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class _LocalProgram:
    @func
    def main(source: Tensor[(8,), "f32"]):
        with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
            sharded = tf.reshard(source, (8 @ thread.lane,), "rmem")
            return tf.local(sharded)

    @func
    def symbolic(source: Tensor[(_N,), "f32", ShardLayout(Layout((_N,), (1,)), (Split(0),), Mesh((Topology("thread", 4),), Layout((4,), (1,)))), "rmem"]):
        return tf.local(source)
