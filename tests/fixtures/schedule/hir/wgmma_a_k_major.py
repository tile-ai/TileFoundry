"""Run a two-stage 64x32 FP8 WGMMA with K-major A and B descriptors.

FP8 admits only K-major descriptors, so ``dtype="fp8e4m3"`` implies ``a_major``
and ``b_major``. Each k iteration copies A's (BK, M) window into a compact
(M, BK) gmem tile before TMA; B is the (K, N) view of a row-major (N, K)
weight, so TMA reads its window with K at stride one. One issue reads 32 bytes
of K: the 64x32 A and 32x32 B fragments each use two core-matrix offsets.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
BK = 32
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_A_K_MAJOR:
    @func
    def gemm(
        at: Tensor[(K, M), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", ((K, N), (1, K))],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 4), (1, 8, 16, 64, 128, 512)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        a = tf.transpose(at[k, :], (1, 0))
                        lhs = tf.schedule(
                            (a,),
                            op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1)))),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16)))),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
