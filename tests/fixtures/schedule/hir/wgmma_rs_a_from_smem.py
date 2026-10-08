"""Read RS-form FP8 WGMMA A after staging it through shared memory.

TMA first stages A like any shared operand. A participant-wise ``T.copy`` then
lands it in the 64x32 FP8 register fragment, the one arrangement accepted by
the RS atom, on the warpgroup that will multiply it: each thread holds four
bytes of K per pair of rows, as a 16-bit fragment does. B stays in shared
memory and must be K-major, so it is the (K, N) view of a row-major (N, K)
weight. This contrasts with the sibling fixture where a preceding WGMMA
accumulator supplies those registers.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
BK = 32
STAGES = 2

_COMPUTE = ThreadMesh((Topology("thread", 256),),
                      ComposedLayout(None, 128, Layout((4, 8, 4), (32, 4, 1))),
                      ("warp", "lane8", "lane4"))
_HELD = (Split(2), Split(0), Split(4))

A_SMEM = Layout(((8, 8), (2, 16)), ((256, 16), (128, 1)))
B_SMEM = Layout(((2, 16), (4, 8)), ((128, 1), (256, 16)))
WEIGHT_VIEW = Layout((K, N), (1, K))

A_REG = ShardLayout(Layout((8, 2, 4, 4, 4, 2), (1, 8, 16, 64, 256, 1024)),
                    _HELD, _COMPUTE)
ACC = ShardLayout(Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)),
                  _HELD, _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_RS_A_FROM_SMEM:
    @func
    def gemm(
        a: Tensor[(M, K), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", WEIGHT_VIEW],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                register = T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.RS)

                with threads[1, :] as _compute:
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

                    with threads[1, :] as _compute:
                        frag = tf.schedule(
                            (lhs,),
                            op=T.copy(rmem_layout=A_REG),
                        )
                        acc = tf.schedule(
                            (acc, frag, rhs),
                            op=T.tiled_mma(atom=register),
                        )

                with threads[1, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
