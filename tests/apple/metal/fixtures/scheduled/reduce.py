"""Two SIMD groups reduce i32 lanes (sum, max) and u32 lanes (max), and resolve a Partial(sum)."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def reduce(
        x: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        wide: Tensor[(2, 32), "u32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        partial: Tensor[(2, 1), "f32", (2 @ simd.g, 1) @ {simd.l @ P("sum")}, "rmem"],
    ):
        with simd:
            total = tf.schedule(
                (tf.empty(Tensor[(2, 1), "i32", (2 @ simd.g, 1), "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.SUM, dtype=DType.i32),
            )
            largest = tf.schedule(
                (tf.empty(Tensor[(2, 1), "i32", (2 @ simd.g, 1), "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.MAX, dtype=DType.i32),
            )
            wide_largest = tf.schedule(
                (tf.empty(Tensor[(2, 1), "u32", (2 @ simd.g, 1), "rmem"]), wide),
                op=T.metal.reduce(kind=T.metal.ReduceKind.MAX, dtype=DType.u32),
            )
            resolved = tf.schedule((tf.empty(Tensor[(2, 1), "f32", (2 @ simd.g, 1), "rmem"]), partial), op=T.metal.reduce())
            return total, largest, wide_largest, resolved
