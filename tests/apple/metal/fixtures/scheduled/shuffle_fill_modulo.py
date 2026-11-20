"""Quad groups of four lanes cannot shift runs of 8 lanes; verification refuses modulo > width."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="shuffle_fill_modulo", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    quads = Mesh(("thread",), layout=Layout(((16, 4),), ((4, 1),)), names=("q", "l"))

    @func
    def shuffle_fill_modulo(
        x: Tensor[(16, 4), "i32", (16 @ quads.q, 4 @ quads.l), "rmem"],
        delta: Tensor[(), "u16", () @ quads, "rmem"],
    ):
        with quads:
            return tf.schedule(
                (tf.empty(Tensor[(16, 4), "i32", (16 @ quads.q, 4 @ quads.l), "rmem"]), x, x, delta),
                op=T.metal.shuffle_fill(kind=T.metal.ShuffleKind.UP, modulo=8, dtype=DType.i32),
            )
