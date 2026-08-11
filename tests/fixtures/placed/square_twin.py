"""A placed square, its faithful reference, and a deliberately drifted reference."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CpuTarget


@module(entry="main", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Model:
    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")

    @func
    def zeroed(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            nothing = tf.sub(x_local, x_local)
            return tf.reshard(nothing, (168 @ cta.block,), "gmem")


@runtime_module(Model)
class Twin:
    @runtime_func
    def main(self, x):
        return x * x

    @runtime_func
    def zeroed(self, x):
        return x - x


@runtime_module(Model)
class Drifted:
    """A deliberate negative whose main result differs from the authored Module."""

    @runtime_func
    def main(self, x):
        return x * x + 0.5

    @runtime_func
    def zeroed(self, x):
        return x - x
