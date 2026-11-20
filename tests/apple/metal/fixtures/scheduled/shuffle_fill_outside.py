"""Two SIMD groups shift each run of 8 lanes up by 9 lanes; evaluation refuses it."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="shuffle_fill_outside", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def shuffle_fill_outside(
        x: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        fill: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        delta: Tensor[(), "u16", () @ simd, "rmem"],
    ):
        with simd:
            return tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"]), x, fill, delta),
                op=T.metal.shuffle_fill(kind=T.metal.ShuffleKind.UP, modulo=8, dtype=DType.i32),
            )
