"""A placed square-subtract expression and its runtime reference."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CpuTarget


@module(entry="fused", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Fused:
    @func
    def fused(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            shifted = tf.sub(squared, x_local)
            return tf.reshard(shifted, (168 @ cta.block,), "gmem")


@runtime_module(Fused)
class FusedTwin:
    @runtime_func
    def fused(self, x):
        return x * x - x
