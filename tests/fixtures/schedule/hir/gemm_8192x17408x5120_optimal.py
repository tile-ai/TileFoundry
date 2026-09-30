"""State the structure of AtomSched's best H200 GEMM kernel.

The 132 persistent CTAs traverse groups of 16 M tiles in N-major order. A
producer warp fills a three-stage SW128 TMA ring while two consumer warpgroups
issue SS WGMMA, then stage each result in separate SW128 memory for a TMA store.
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
STAGES = 3
CTAS = 132
GM = M // BM
GN = N // BN
GROUP_M = 16

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
    topologies=(Topology("cta", CTAS), Topology("thread", 384)),
)
class GEMM_8192X17408X5120_OPTIMAL:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for g in range(GM // GROUP_M):
                    for bn in range(GN):
                        start = (cta.persistent - (g * GN + bn) * GROUP_M) % CTAS
                        for mi in range(start, GROUP_M, CTAS):
                            m = (g * GROUP_M + mi) * BM
                            n = bn * BN
                            with threads[1:3, :] as _compute:
                                acc = tf.zeros(Tensor[(BM, BN), "f32", ACC, "rmem"])

                            for k in tile(K, BK):
                                with threads[0, :32] as _loader:
                                    lhs = tf.schedule(
                                        (a[m:m + BM, k],),
                                        op=T.copy_async_tensor(smem_layout=A_SMEM),
                                        buffers=STAGES,
                                    )
                                    rhs = tf.schedule(
                                        (b[k, n:n + BN],),
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
                                staged = tf.schedule(
                                    (tile_out,), op=T.copy(smem_layout=OUT_SMEM)
                                )

                            with threads[0, :32] as _storer:
                                tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                out = tf.insert_slice(out, tile, (m, n))

                return out
