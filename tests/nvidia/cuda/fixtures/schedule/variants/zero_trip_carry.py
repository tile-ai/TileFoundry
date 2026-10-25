from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ZeroTripCarry:
    @func
    def gemm(x: Tensor[(4,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:  # noqa: F841
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:  # noqa: F841
                acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                for k in range(0):
                    acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                return tf.schedule((acc, tf.empty(Tensor[(4,), "f32"])), op=T.copy())
