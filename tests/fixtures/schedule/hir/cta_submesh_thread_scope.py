"""A thread scope inside a CTA submesh retains the same thread positions."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 4), Topology("thread", 128)))
class CtaSubmeshThreadScope:
    @func
    def gemm(x: Tensor[(4, 32), "f32"]) -> Tensor[(4, 32), "f32"]:
        with Mesh(("cta",), layout=(2, 2), names=("b", "c")) as cta:
            with cta[:, :1] as _part:
                with Mesh(("thread",), layout=(4, 32), names=("w", "t")) as threads:
                    held = tf.schedule((x, tf.empty(Tensor[(4, 32), "f32", (4 @ threads.w, 32 @ threads.t), "rmem"])), op=T.copy())
                    return tf.schedule((held, tf.empty(Tensor[(4, 32), "f32"])), op=T.copy())
