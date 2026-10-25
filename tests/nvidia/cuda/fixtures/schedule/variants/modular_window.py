from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ModularWindow:
    @func
    def gemm(x: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:  # noqa: F841
                out = tf.zeros(Tensor[(8,), "f32", "rmem"])
                for u in range(cta.b, 8, 1):
                    start = (u % 4) * 8
                    out = tf.schedule((x[start:start+8], tf.empty(Tensor[(8,), "f32", "rmem"])), op=T.copy())
                return out
