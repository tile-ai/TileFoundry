"""Load both register operands of an SM80 warp MMA from shared memory.

The warp collectively issues ``T.ldmatrix`` for A, whose destination fragment
is fixed by the MMA declaration. Each participant uses ``T.copy`` for B, whose
register layout is authored explicitly. The compute mesh is the second warp;
the staged A and B tiles retain separate shared-memory arrangements.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 16
N = 8
K = 32
BK = 16
STAGES = 2

_COMPUTE = ThreadMesh((Topology("thread", 64),),
                      ComposedLayout(None, 32, Layout((4, 8), (1, 4))),
                      ("warp", "lane"))

A_SMEM = Layout((16, 16), (16, 1))
B_SMEM = Layout((16, 8), (8, 1))

B_REG = ShardLayout(Layout((8, 2, 4, 2), (1, 8, 16, 64)),
                    (Split(2), Split(0)), _COMPUTE)
ACC = ShardLayout(Layout((2, 4, 8, 2), (1, 2, 8, 64)),
                  (Split(1), Split(2)), _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 64)),
)
class SM80_MMA_LDMATRIX:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 32),
                names=("role", "participant"),
            ) as threads:
                warp = T.cuda.sm80.Mma()

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
                        lhs_frag = tf.schedule((lhs,), op=T.ldmatrix())
                        rhs_frag = tf.schedule((rhs,), op=T.copy(rmem_layout=B_REG))
                        acc = tf.schedule(
                            (acc, lhs_frag, rhs_frag),
                            op=T.tiled_mma(atom=warp),
                        )

                with threads[1, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
