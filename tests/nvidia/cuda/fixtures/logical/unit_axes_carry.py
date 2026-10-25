from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="accumulate", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2),))
class UnitAxesCarry:
    @func
    def accumulate(x: Tensor[(1, 1, 64), "f32"]):
        with Mesh(("cta",), layout=(1, 1, 2), names=("a", "b", "c")) as cta:
            total = tf.zeros(Tensor[(1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "f32", "rmem"])
            for i in range(2):
                part = tf.reshard(x, (1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "rmem")
                total = total + part
            return tf.reshard(total, (1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "gmem")
