"""A placed weighted product and references for child and standalone use.

Weighted has no target for use as a child. WeightedRoot supplies cpu.Target for
standalone checks because only a root may declare one.
"""

from dataclasses import replace

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="scaled", topologies=(Topology("cta", 168),))
class Weighted:
    @func
    def scaled(
        x: Tensor[(168,), "f32"], w: ConstTensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            w_local = tf.reshard(w, (168 @ cta.block,), "rmem")
            weighted = tf.mul(x_local, w_local)
            return tf.reshard(weighted, (168 @ cta.block,), "gmem")


@runtime.module(Weighted)
class WeightedTwin:
    @runtime.func
    def scaled(self, x, w):
        return x * w


WeightedRoot = replace(Weighted, target=cpu.Target())


@runtime.module(WeightedRoot)
class WeightedRootTwin:
    @runtime.func
    def scaled(self, x, w):
        return x * w
