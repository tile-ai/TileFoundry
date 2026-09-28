"""A 256-wide schedule whose authored order makes N the outer atom loop.

The work and instruction are the same as ``wgmma_repeat_along_n``.  Only the
MMA order differs: each warpgroup walks N outside M, with K innermost.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 128
N = 256
K = 32
BK = 16
STAGES = 2


_COMPUTE = ThreadMesh((Topology("thread", 384),),
                      ComposedLayout(None, 128, Layout((2, 4, 8, 4), (128, 32, 4, 1))),
                      ("group", "warp", "lane8", "lane4"))
A_SMEM = Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1)))
B_SMEM = Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1)))
ACC = ShardLayout(Layout((2, 4, 8, 2, 4, 2, 4, 8),
                         (16384, 4096, 1, 8, 16, 64, 128, 512)),
                  (Split(0), Split(4), Split(2), Split(6)), _COMPUTE)


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
                    n=64, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with threads[1:3, :] as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])

                for k in tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=A_SMEM),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=B_SMEM),
                            buffers=STAGES,
                        )

                    with threads[1:3, :] as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                            repeat=(2, 4, 1),
                            order=(1, 0, 2),
                        )

                with threads[1:3, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
