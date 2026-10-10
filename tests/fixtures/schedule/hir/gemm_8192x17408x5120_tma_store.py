"""Store the winning three-stage schedule through shared memory with TMA.

A and B each occupy one SW128 TMA box. A ``T.copy`` stages the finished tile
in N-contiguous SW128 rows, then a reverse-direction TMA writes that box to
global memory. Its result already has the output window's storage and layout,
so the final ``tf.insert_slice`` identifies the destination without a copy.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
STAGES = 3


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class GEMM_8192X17408X5120_TMA_STORE:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for m in tf.tile(M, BM):
                    for n in tf.tile(N, BN):
                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                            acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])

                        for k in tf.tile(K, BK):
                            with threads[0, :32] as _loader:
                                lhs = tf.schedule(
                                    (a[m, k],),
                                    op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)),
                                    buffers=STAGES,
                                )
                                rhs = tf.schedule(
                                    (b[k, n],),
                                    op=T.copy_async_tensor(smem_layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)),
                                    buffers=STAGES,
                                )

                            with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                                acc = tf.schedule(
                                    (acc, lhs, rhs),
                                    op=T.tiled_mma(atom=wgmma),
                                    repeat=(2, 1, 4),
                                )

                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                            tile_out = tf.cast(acc, dtype="bf16")
                            staged = tf.schedule((tile_out,), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))

                        with threads[0, :32] as _storer:
                            tile = tf.schedule((staged,), op=T.copy_async_tensor())
                            out = tf.insert_slice(out, tile, (m, n))
                return out
