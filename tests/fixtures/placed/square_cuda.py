"""A CTA-sharded elementwise square for a CUDA target."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 168),))
class Model:
    @func
    def main(x: Tensor[(168,), "f32"]):
        with Mesh(("cta",), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")
