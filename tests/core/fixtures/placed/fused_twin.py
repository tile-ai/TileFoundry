"""A placed square-subtract expression and its runtime reference."""

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="fused", target=cpu.Target(), topologies=(Topology("cta", 168),))
class Fused:
    @func
    def fused(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            shifted = tf.sub(squared, x_local)
            return tf.reshard(shifted, (168 @ cta.block,), "gmem")


@runtime.module(Fused)
class FusedTwin:
    @runtime.func
    def fused(self, x):
        return x * x - x
