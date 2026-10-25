from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 16)))
class RegisterRows:
    @func
    def gemm(x: Tensor[(16,16), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:  # noqa: F841
            with Mesh(("thread",), layout=(16,), names=("col",)) as th:
                held = tf.schedule((x, tf.empty(Tensor[(16,16), "f32", (16,16 @ th.col), "rmem"])), op=T.copy())
                out = tf.zeros(Tensor[(16,16), "f32"])
                for i in range(16):
                    out = tf.insert_slice(out, held[i:i+1, :], (i,0))
                return out
