"""Two SIMD groups vote on a predicate; bit t of the result is lane t's vote."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="ballot", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def ballot(even: Tensor[(2, 32), "bool", (2 @ simd.g, 32 @ simd.l), "rmem"]):
        with simd:
            return tf.schedule((tf.empty(Tensor[(2, 1), "u32", (2 @ simd.g, 1), "rmem"]), even), op=T.metal.ballot())
