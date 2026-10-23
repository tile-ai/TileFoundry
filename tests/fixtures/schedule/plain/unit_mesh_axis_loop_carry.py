"""Keep a split on an extent-one mesh axis through a loop carry."""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

B = 1
N = 128
D = 64


@module(entry="acc", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
class UnitAxisCarry:
    @func
    def acc(x: Tensor[(B, N, D), "f32"]):
        with Mesh(("cta",), layout=(B, 2), names=("b", "n")) as cta:
            total = tf.zeros(Tensor[(B @ cta.b, N @ cta.n, D), "f32", "rmem"])
            for t in tf.tile(D, 32):
                part = tf.reshard(x, (B @ cta.b, N @ cta.n, D), "rmem")
                total = total + tf.reshard(part, (B @ cta.b, N @ cta.n, D), "rmem")
            return tf.reshard(total, (B @ cta.b, N @ cta.n, D), "gmem")
