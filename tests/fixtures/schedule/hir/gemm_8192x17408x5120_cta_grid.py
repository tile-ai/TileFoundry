"""Deal the winning schedule's tiles across a 64 by 68 CTA grid.

Each CTA runs one output tile at its ``(bm, bn)`` coordinate. Per-CTA staging
matches the single-CTA TMA-store fixture: A and B use one SW128 box apiece,
and the finished tile uses N-contiguous SW128 rows before its TMA store.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 8192
K = 5120
N = 17408
BM = 128
BN = 256
BK = 64
STAGES = 4
GM = M // BM
GN = N // BN

_COMPUTE = ThreadMesh((Topology("thread", 384),),
                      ComposedLayout(None, 128, Layout((2, 4, 8, 4), (128, 32, 4, 1))),
                      ("group", "warp", "lane8", "lane4"))
A_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0,
                        Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))))
B_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0,
                        Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))))
OUT_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0, Layout((128, (4, 64)), (64, (8192, 1))))
ACC = ShardLayout(Layout((2, 8, 2, 4, 2, 4, 32),
                         (16384, 1, 8, 16, 64, 128, 512)),
                  (Split(0), Split(3), Split(1), Split(5)), _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", GM * GN), Topology("thread", 384)),
)
class GEMM_8192X17408X5120_CTA_GRID:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GM, GN), names=("bm", "bn")) as cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for m in tile(cta.bm * BM, (cta.bm + 1) * BM, BM):
                    for n in tile(cta.bn * BN, (cta.bn + 1) * BN, BN):
                        with threads[1:3, :] as _compute:
                            acc = tf.zeros(Tensor[(BM, BN), "f32", ACC, "rmem"])

                        for k in tile(K, BK):
                            with threads[0, :32] as _loader:
                                lhs = tf.schedule(
                                    (a[m, k],),
                                    op=T.copy_async_tensor(smem_layout=A_SMEM),
                                    buffers=STAGES,
                                )
                                rhs = tf.schedule(
                                    (b[k, n],),
                                    op=T.copy_async_tensor(smem_layout=B_SMEM),
                                    buffers=STAGES,
                                )

                            with threads[1:3, :] as _compute:
                                acc = tf.schedule(
                                    (acc, lhs, rhs),
                                    op=T.tiled_mma(atom=wgmma),
                                    repeat=(2, 1, 4),
                                )

                        with threads[1:3, :] as _compute:
                            tile_out = tf.cast(acc, dtype="bf16")
                            staged = tf.schedule((tile_out,), op=T.copy(smem_layout=OUT_SMEM))

                        with threads[0, :32] as _storer:
                            tile = tf.schedule((staged,), op=T.copy_async_tensor())
                            out = tf.insert_slice(out, tile, (m, n))
                return out
