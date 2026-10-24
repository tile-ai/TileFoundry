"""Persistent split-K work rasterized in groups of 33 across 128 CTAs.

Each CTA writes its K-piece into a separate workspace band. The first wave's
weight tiles exercise the union of mesh-parameter fibers without quantifiers.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

M = 4224
K = 512
N = 2048
BM = 128
BN = 256
BK = 64
STAGES = 2
CTAS = 128
PIECES = 4
GROUP = 33
PIECE_TILES = K // (PIECES * BK)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", CTAS), Topology("thread", 384)),
)
class PersistentTileUnionCount:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(PIECES * M, N), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(PIECES * M, N), "bf16"])
                for work in range(_cta.block, (M // BM) * (N // BN) * PIECES, CTAS):
                    tile = work // PIECES
                    piece = work % PIECES
                    m = (tile % GROUP) * BM
                    n = (tile // GROUP) * BN
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 32, 4 @ _compute.lane4, 2,), (16384, 16, 8, 1, 512, 128, 64)), "rmem"])

                    for kk in range(PIECE_TILES):
                        k = (piece * PIECE_TILES + kk) * BK
                        with threads[0, :32] as _loader:
                            lhs_ring = tf.empty(
                                Tensor[
                                    (128, 64), "bf16",
                                    Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    "smem",
                                ],
                                buffers=STAGES,
                            )
                            lhs = tf.schedule(
                                (a[m:m + BM, k:k + BK], lhs_ring),
                                op=T.copy_async_tensor(),
                            )
                            rhs_ring = tf.empty(
                                Tensor[
                                    (64, 256), "bf16",
                                    Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    "smem",
                                ],
                                buffers=STAGES,
                            )
                            rhs = tf.schedule(
                                (b[k:k + BK, n:n + BN], rhs_ring),
                                op=T.copy_async_tensor(),
                            )

                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                            acc = tf.schedule(
                                (acc, lhs, rhs),
                                op=T.tiled_mma(atom=wgmma),
                                repeat=(2, 1, 4),
                            )

                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        tile_out = tf.cast(acc, dtype="bf16")
                        staged_dst = tf.empty(
                            Tensor[
                                (128, 256), "bf16",
                                Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                "smem",
                            ],
                        )
                        staged = tf.schedule((tile_out, staged_dst), op=T.copy())

                    with threads[0, :32] as _storer:
                        tile_dst = tf.empty(Tensor[(128, 256), "bf16", (128, 256)])
                        tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (piece * M + m, n))
                return out
