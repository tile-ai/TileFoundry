"""Two SIMD groups rotate their lanes down by one, broadcast lane 5, and rotate u32 lanes up."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="shuffle", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def shuffle(
        x: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        wide: Tensor[(2, 32), "u32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        one: Tensor[(), "u16", () @ simd, "rmem"],
        five: Tensor[(), "u16", () @ simd, "rmem"],
    ):
        with simd:
            down = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"]), x, one),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.ROTATE_DOWN, dtype=DType.i32),
            )
            broadcast = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"]), x, five),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.BROADCAST, dtype=DType.i32),
            )
            up = tf.schedule(
                (tf.empty(Tensor[(2, 32), "u32", (2 @ simd.g, 32 @ simd.l), "rmem"]), wide, one),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.ROTATE_UP, dtype=DType.u32),
            )
            return down, broadcast, up
