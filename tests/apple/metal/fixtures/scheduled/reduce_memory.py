"""Two SIMD groups cannot reduce lanes held in device memory; verification refuses the source."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="reduce_memory", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def reduce_memory(x: Tensor[(2, 32), "i32", ((2, 32), (32, 1)), "gmem"]):
        with simd:
            return tf.schedule(
                (tf.empty(Tensor[(2, 1), "i32", (2 @ simd.g, 1), "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.SUM, dtype=DType.i32),
            )
