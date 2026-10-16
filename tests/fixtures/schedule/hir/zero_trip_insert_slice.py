"""Keep a zero-trip loop while giving its InsertSlice the destination's address."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 1)))
class ZeroTripInsertSlice:
    @func
    def gemm(x: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _thread:
                out = tf.zeros(Tensor[(2, 4), "f32"])
                for _i in range(0):
                    row = tf.schedule((x, tf.empty(Tensor[(1, 4), "f32", (1, 4), "rmem"])), op=T.copy())
                    out = tf.insert_slice(out, row, (0, 0))
                return out
