"""Load both register operands of an SM80 warp MMA from shared memory.

The warp collectively issues ``T.ldmatrix`` for A, whose destination fragment
is fixed by the MMA declaration. Each participant uses ``T.copy`` for B, whose
register layout is authored explicitly. The compute mesh is the second warp;
the staged A and B tiles retain separate shared-memory arrangements.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 16
N = 8
K = 32
BK = 16
STAGES = 2


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
        a_smem = Layout((16, 16), (16, 1))
        b_smem = Layout((16, 8), (8, 1))
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 32),
                names=("role", "participant"),
            ) as threads:
                warp = T.cuda.sm80.Mma()

                with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((2, 4 @ _compute.warp, 8 @ _compute.lane, 2), (1, 2, 8, 64)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=a_smem),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=b_smem),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                        lhs_frag = tf.schedule((lhs,), op=T.ldmatrix())
                        rhs_frag = tf.schedule((rhs,), op=T.copy(rmem_layout=((8 @ _compute.lane, 2, 4 @ _compute.warp, 2), (1, 8, 16, 64))))
                        acc = tf.schedule(
                            (acc, lhs_frag, rhs_frag),
                            op=T.tiled_mma(atom=warp),
                        )

                with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
