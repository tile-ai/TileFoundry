"""Reject a destination whose authored positions cross tensor axes."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 256)))
class LayoutCrossingAxes:
    @func
    def gemm(x: Tensor[(64, 128), "f16"]) -> Tensor[(64, 128), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(2, 8, 4, 4), names=("g", "lane8", "warp", "lane4")) as threads:
                dst = tf.empty(Tensor[(64, 128), "f16", ((2 @ threads.g, 8 @ threads.lane8, 2, 4 @ threads.warp, 2, 4 @ threads.lane4, 8), (4096, 1, 8, 16, 64, 128, 512)), "rmem"])
                held = tf.schedule((x, dst), op=T.copy())
                return tf.cast(held, dtype="f32")
