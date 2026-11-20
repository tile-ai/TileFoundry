"""Every lane of two SIMD groups names lane 40 as its source; evaluation refuses it."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="shuffle_outside", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def shuffle_outside(
        x: Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"],
        lane: Tensor[(2, 32), "u16", (2 @ simd.g, 32 @ simd.l), "rmem"],
    ):
        with simd:
            return tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", (2 @ simd.g, 32 @ simd.l), "rmem"]), x, lane),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.INDEX, dtype=DType.i32),
            )
