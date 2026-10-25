from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class WrittenReturnWindow:
    @func
    def gemm(x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as cta:  # noqa: F841
            out = tf.zeros(Tensor[(2, 4), "f32"])
            updated = tf.insert_slice(out, x, (1, 0))
            return updated[1:2, :]
