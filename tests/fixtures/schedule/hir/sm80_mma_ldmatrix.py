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
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 32),
                names=("role", "participant"),
            ) as threads:
                warp = T.cuda.sm80.Mma()

                with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((2, 8 @ _compute.lane, 4 @ _compute.warp, 2), (64, 8, 2, 1)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(
                            Tensor[
                                (16, 16), "bf16",
                                (16, 16),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        lhs = tf.schedule(
                            (a[:, k], lhs_ring),
                            op=T.copy_async_tensor(),
                        )
                        rhs_ring = tf.empty(
                            Tensor[
                                (16, 8), "bf16",
                                (16, 8),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :], rhs_ring),
                            op=T.copy_async_tensor(),
                        )

                    with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                        lhs_frag_dst = tf.empty(
                            Tensor[
                                (16, 16), "bf16",
                                ShardLayout(
                                    layout=Layout((2, 8, 2, 4, 2), (128, 16, 8, 2, 1)),
                                    attrs=(S(3), S(1)),
                                    mesh=_compute,
                                ),
                                "rmem",
                            ],
                        )
                        lhs_frag = tf.schedule((lhs, lhs_frag_dst), op=T.ldmatrix())
                        rhs_frag_dst = tf.empty(
                            Tensor[
                                (16, 8), "bf16",
                                ShardLayout(
                                    layout=Layout((2, 4, 2, 8), (64, 16, 8, 1)),
                                    attrs=(S(1), S(3)),
                                    mesh=_compute,
                                ),
                                "rmem",
                            ],
                        )
                        rhs_frag = tf.schedule((rhs, rhs_frag_dst), op=T.copy())
                        acc = tf.schedule(
                            (acc, lhs_frag, rhs_frag),
                            op=T.tiled_mma(atom=warp),
                        )

                output = tf.empty(Tensor[(M, N), "bf16"])
                with Mesh(threads[1, :], layout=((4, 8), (1, 4)), names=("warp", "lane")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                    result = tf.insert_slice(output, result, (0, 0))
                return result
