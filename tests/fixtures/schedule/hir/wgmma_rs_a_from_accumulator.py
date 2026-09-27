"""Read RS-form WGMMA A from a preceding WGMMA accumulator.

The first 64x16 product and the second atom's A share one CLayout_64x16
register arrangement. ``tf.cast`` narrows the first result to bf16 without
changing its layout or storage, so ``p = a @ b0`` feeds ``p @ b1`` without
leaving the issuing warpgroup's registers.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 64
K = 32
N0 = 16
N1 = 32
BK = 16
STAGES = 2

_COMPUTE = ThreadMesh((Topology("thread", 256),),
                      ComposedLayout(None, 128, Layout((4, 8, 4), (32, 4, 1))),
                      ("warp", "lane8", "lane4"))
A_SMEM = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
B0_SMEM = Layout(((2, 8), (2, 8)), ((64, 8), (128, 1)))
B1_SMEM = Layout(((2, 8), (4, 8)), ((64, 8), (128, 1)))
_HELD = (Split(2), Split(0), Split(4))

P_REG = ShardLayout(Layout((8, 2, 4, 2, 4, 2), (1, 8, 16, 64, 128, 512)),
                    _HELD, _COMPUTE)
ACC = ShardLayout(Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)),
                  _HELD, _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_RS_A_FROM_ACCUMULATOR:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N0), "bf16"],
        b1: Tensor[(N0, N1), "bf16"],
    ) -> Tensor[(M, N1), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                shared = T.cuda.sm90.Wgmma(
                    n=16, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)
                register = T.cuda.sm90.Wgmma(n=32, form=T.cuda.sm90.Form.RS)

                with threads[1, :] as _compute:
                    p = tf.zeros(Tensor[(M, N0), "f32", P_REG, "rmem"])

                for k in tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=A_SMEM),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=B0_SMEM),
                            buffers=STAGES,
                        )

                    with threads[1, :] as _compute:
                        p = tf.schedule(
                            (p, lhs, rhs),
                            op=T.tiled_mma(atom=shared),
                        )

                with threads[1, :] as _compute:
                    value = tf.cast(p, dtype="bf16")
                    acc = tf.zeros(Tensor[(M, N1), "f32", ACC, "rmem"])

                for j in tile(N0, BK):
                    with threads[0, :32] as _loader:
                        rhs = tf.schedule(
                            (b1[j, :],),
                            op=T.copy_async_tensor(smem_layout=B1_SMEM),
                            buffers=STAGES,
                        )

                    with threads[1, :] as _compute:
                        acc = tf.schedule(
                            (acc, value, rhs),
                            op=T.tiled_mma(atom=register),
                        )

                with threads[1, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
