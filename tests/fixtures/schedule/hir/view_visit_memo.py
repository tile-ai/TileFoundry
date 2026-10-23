"""Keep authored identity views while forwarding region operands without wrappers."""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ViewVisitMemo:
    @func
    def gemm(x: Tensor[(16, 8), "f32"], index: Tensor[(1,), "i64"]) -> Tensor[(1, 8), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _block:
            with Mesh(("thread",), layout=(1,), names=("lane",)) as _lane:
                row = index[0]
                identity = tf.reshape(x, (16, 8))
                window = identity[row:row + 1, :]
                return tf.schedule((window, tf.empty(Tensor[(1, 8), "f32", "rmem"])), op=T.copy())
