"""Issue n16 atoms within each warpgroup share of a split N axis."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

Q_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), (3, 4, 16)), ((512, 64), (4096, 16, 1))))
K_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((12, 8), (3, 4, 16)), ((512, 64), (6144, 16, 1))))
K_T16 = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((3, 4, 16), (2, 3, 2, 8)), ((6144, 16, 1), (3072, 1024, 512, 64))))


def _wg(n):
    return T.cuda.sm90.Wgmma(n=n, dtype="bf16", form=T.cuda.sm90.Form.SS,
                             a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 384)))
class N16:
    @func
    def gemm(q: Tensor[(64, 192), "bf16"], k: Tensor[(96, 192), "bf16"]) -> Tensor[(64, 96), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                with threads[0, :32] as _loader:
                    q_s = tf.schedule((q, tf.empty(Tensor[(64, 192), "bf16", Q_SMEM, "smem"])), op=T.copy_async_tensor())
                    k_s = tf.schedule((k, tf.empty(Tensor[(96, 192), "bf16", K_SMEM, "smem"])), op=T.copy_async_tensor())
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as cg:
                    s = tf.schedule(
                        (tf.zeros(Tensor[(64, 96), "f32", ((4 @ cg.warp, 2, 8 @ cg.lane8, 2 @ cg.group, 3, 2, 4 @ cg.lane4, 2), (16, 8, 1, 3072, 1024, 512, 128, 64)), "rmem"]),
                         q_s, tf.bitcast(k_s, K_T16)),
                        op=T.tiled_mma(atom=_wg(16)),
                    )
                    return tf.schedule((s, tf.empty(Tensor[(64, 96), "f32", (64, 96)])), op=T.copy())
