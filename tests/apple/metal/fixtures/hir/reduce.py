"""Two SIMD groups combine a per-lane Partial(sum); Metal has no bf16 SIMD reduction."""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="hir_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    simd = Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),)), names=("g", "l"))

    @func
    def hir_reduce(
        single: Tensor[(2, 1), "f32", (2 @ simd.g, 1) @ {simd.l @ P("sum")}, "rmem"],
        brain: Tensor[(2, 1), "bf16", (2 @ simd.g, 1) @ {simd.l @ P("sum")}, "rmem"],
    ):
        with simd:
            total = tf.reshard(single, (2 @ simd.g, 1), "rmem")
            brain_total = tf.reshard(brain, (2 @ simd.g, 1), "rmem")
            return total, brain_total
