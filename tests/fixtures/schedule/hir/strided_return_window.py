"""Keep a cropped 24x512 row's 512 stride when returning 24x256 elements."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 32), Topology("thread", 1)))
class StridedReturnWindow:
    @func
    def gemm(x: Tensor[(32, 14336), "bf16"]) -> Tensor[(24, 256), "bf16", "rmem"]:
        with Mesh(("cta",), layout=(32,), names=("t",)) as cta:
            row = x[cta.t:cta.t + 1, 0:12288]
            matrix = tf.reshape(row, new_shape=(24, 512))
            window = matrix[:, :256]
            with Mesh(("thread",), layout=(1,), names=("t",)) as _thread:
                return tf.schedule((window, tf.empty(Tensor[(24, 256), "bf16", (24, 256), "rmem"])), op=T.copy())
