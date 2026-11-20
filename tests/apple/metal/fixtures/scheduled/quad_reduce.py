"""Sixteen quad groups of four lanes each sum their lanes."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="quad_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    quads = Mesh(("thread",), layout=Layout(((16, 4),), ((4, 1),)), names=("q", "l"))

    @func
    def quad_reduce(x: Tensor[(16, 4), "i32", (16 @ quads.q, 4 @ quads.l), "rmem"]):
        with quads:
            return tf.schedule(
                (tf.empty(Tensor[(16, 1), "i32", (16 @ quads.q, 1), "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.SUM, dtype=DType.i32),
            )
