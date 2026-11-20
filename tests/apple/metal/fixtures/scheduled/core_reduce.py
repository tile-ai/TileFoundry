"""Two SIMD groups select core's generic T.reduce, which Metal does not lower."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="core_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def core_reduce(x: Tensor[(2, 32), "f32", (2 @ simd.g, 32 @ simd.l), "rmem"]):
        with simd:
            total = tf.empty(Tensor[(2, 1), "f32", (2 @ simd.g, 1), "rmem"])
            return tf.schedule((x, total), op=T.reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM))
