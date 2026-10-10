"""Read an explicitly M-contiguous A with a swizzled MN-major descriptor.

The caller supplies ``a`` as an (M, K) tensor with M contiguous.
TMA lands A's 64-element
M run as one SW128 row, while B uses a 64-byte N-contiguous row. This isolates
the atom's major-mode choice from the rest of the 64x32 schedule.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 32
BK = 16
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_A_MN_MAJOR:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16", ((M, K), (1, M)), "gmem"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 4), (1, 8, 16, 64, 128, 512)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 8), (1, 32)), ((256, 32), (512, 1))) | Swizzle(2, 4, 3)),
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
