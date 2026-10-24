"""Round-trip a scalar ceildiv whose numerator is a mesh coordinate."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

NB = 2


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 4), Topology("thread", 1)))
class ScalarCeildivGate:
    @func
    def gemm(x: Tensor[(64,), "f32"]) -> Tensor[(8,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(4,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _thread:
                gate = ceildiv(cta.b, NB)
                return tf.schedule((x[gate:gate + 8], tf.empty(Tensor[(8,), "f32", (8,), "rmem"])), op=T.copy())
