"""A placed weighted product and references for child and standalone use.

Weighted has no target for use as a child. WeightedRoot supplies CpuTarget for
standalone checks because only a root may declare one.
"""

from dataclasses import replace

from tilefoundry import module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CpuTarget


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


@runtime_module(Weighted)
class WeightedTwin:
    @runtime_func
    def scaled(self, x, w):
        return x * w


WeightedRoot = replace(Weighted, target=CpuTarget())


@runtime_module(WeightedRoot)
class WeightedRootTwin:
    @runtime_func
    def scaled(self, x, w):
        return x * w
