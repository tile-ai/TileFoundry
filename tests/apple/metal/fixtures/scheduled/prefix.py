"""Two SIMD groups take the exclusive prefix sum of i32 lanes and the inclusive one of u32 lanes."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="prefix", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def prefix(
        x: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        wide: Tensor[(2, 32), "u32", (2 @ simd.g, 32 @ simd.l), "rmem"],
    ):
        with simd:
            exclusive = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"]), x),
                op=T.metal.prefix(kind=T.metal.PrefixKind.SUM, inclusive=False, dtype=DType.i32),
            )
            inclusive = tf.schedule(
                (tf.empty(Tensor[(2, 32), "u32", (2 @ simd.g, 32 @ simd.l), "rmem"]), wide),
                op=T.metal.prefix(kind=T.metal.PrefixKind.SUM, inclusive=True, dtype=DType.u32),
            )
            return exclusive, inclusive
