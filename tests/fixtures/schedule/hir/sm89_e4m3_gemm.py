"""Stage E4M3 K slices through shared memory with cp.async for the SM89 warp MMA.

One warp computes the 16 x 8 F32 tile of A (16, 128) by B, the (K, N) view of a
row-major (N, K) weight. Each 32-wide K slice of A lands row-major and each slice
of B K-contiguous; ``T.copy`` gathers them into the SM89 atom's fragments, and the
MMA accumulates. The waits and barriers between the copies and their consumers
belong to the implementation, as in ``tests/fixtures/tir/e4m3_gemm.py``.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

M = 16
N = 8
K = 128
BK = 32


@module(
    entry="gemm",
    target=CudaTarget("nvidia.rtx_6000_ada"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class SM89_E4M3_GEMM:
    @func
    def gemm(
        a: Tensor[(M, K), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", ((K, N), (1, K))],
    ) -> Tensor[(M, N), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=((4, 8), (1, 4)), names=("warp", "lane")) as threads:
                atom = T.cuda.sm89.Mma()
                acc = tf.zeros(Tensor[
                    (M, N), "f32",
                    ((2, 8 @ threads.lane, 4 @ threads.warp, 2), (64, 8, 2, 1)), "rmem",
                ])
                for k in tf.tile(K, BK):
                    lhs_stage = tf.empty(
                        Tensor[(M, BK), "fp8e4m3", Layout((M, BK), (BK, 1)), "smem"]
                    )
                    lhs = tf.schedule((a[:, k], lhs_stage), op=T.copy_async())
                    rhs_stage = tf.empty(
                        Tensor[(BK, N), "fp8e4m3", Layout((BK, N), (1, BK)), "smem"]
                    )
                    rhs = tf.schedule((b[k, :], rhs_stage), op=T.copy_async())
                    lhs_frag_dst = tf.empty(
                        Tensor[
                            (M, BK), "fp8e4m3",
                            ShardLayout(
                                layout=Layout((2, 8, 2, 4, 4), (256, 32, 16, 4, 1)),
                                attrs=(S(3), S(1)),
                                mesh=threads,
                            ),
                            "rmem",
                        ]
                    )
                    lhs_frag = tf.schedule((lhs, lhs_frag_dst), op=T.copy())
                    rhs_frag_dst = tf.empty(
                        Tensor[
                            (BK, N), "fp8e4m3",
                            ShardLayout(
                                layout=Layout((2, 4, 4, 8), (128, 32, 8, 1)),
                                attrs=(S(1), S(3)),
                                mesh=threads,
                            ),
                            "rmem",
                        ]
                    )
                    rhs_frag = tf.schedule((rhs, rhs_frag_dst), op=T.copy())
                    acc = tf.schedule((acc, lhs_frag, rhs_frag), op=T.tiled_mma(atom=atom))
                output = tf.zeros(Tensor[(M, N), "f32"])
                return tf.insert_slice(output, acc, (0, 0))
