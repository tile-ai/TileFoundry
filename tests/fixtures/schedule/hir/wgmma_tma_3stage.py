"""Stage TMA transfers three deep for two 64x16 WGMMA groups.

The shared descriptors tile each 64x16x16 operand over two M groups, and the
register arrangement is held by the corresponding pair of warpgroups.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 16
K = 64
BK = 16
STAGES = 3


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class WGMMA_TMA_3STAGE:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 2), (1024, 1, 8, 16, 64, 128, 512)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1)))),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1)))),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
