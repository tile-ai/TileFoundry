"""Store a narrowed WGMMA accumulator into SW32 shared memory."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

P_SMEM = ComposedLayout(Swizzle(1, 4, 3), 0, Layout(((4, 2, 8), (3, 16)), ((256, 128, 16), (1024, 1))))

@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class StoreP:
    @func
    def gemm(s: Tensor[(64, 48), "f32"]) -> Tensor[(64, 48), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("one",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as g:
                acc = tf.schedule(
                    (s, tf.empty(Tensor[(64, 48), "f32", ((4 @ g.warp, 2, 8 @ g.lane8, 6, 4 @ g.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"])),
                    op=T.copy(),
                )
                p = tf.cast(acc, dtype="bf16")
                p_s = tf.schedule((p, tf.empty(Tensor[(64, 48), "bf16", P_SMEM, "smem"])), op=T.stmatrix())
                return tf.schedule((p_s, tf.empty(Tensor[(64, 48), "bf16", (64, 48)])), op=T.copy())
