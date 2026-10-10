"""A 256-wide schedule whose authored order makes N the outer atom loop.

The N repeat is a true loop. Its authored order makes each warpgroup walk N
outside M, with K innermost.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 256
K = 32
BK = 16
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class WGMMA_REPEAT_ALONG_N_ORDER:
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
                    n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((2 @ _compute.group, 4, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 8), (16384, 4096, 1, 8, 16, 64, 128, 512)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1)))),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1)))),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                            repeat=(2, 4, 1),
                            order=(1, 0, 2),
                        )

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
