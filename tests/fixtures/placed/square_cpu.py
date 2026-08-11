"""A CTA-sharded elementwise square for the CPU target."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CpuTarget


@module(entry="main", target=CpuTarget(), topologies=(Topology("cta", 168),))
class Mine:
    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(("cta",), (168,), ("block",)) as cta:
            local = tf.reshard(x, (168 @ cta.block,), "rmem")
            return tf.reshard(tf.square(local), (168 @ cta.block,), "gmem")
