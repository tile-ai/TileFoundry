"""Stage E4M3 K slices through shared memory with cp.async for the SM89 warp MMA.

One warp computes the 16 x 8 F32 tile of A (16, 128) by B, the (K, N) view of a
row-major (N, K) weight. Each 32-wide K slice is copied by threads that own
disjoint parts of it: 16 threads one row of A each, 8 threads one K-contiguous
column of B each. The whole slices are then read through views of the same shared
memory, ``T.copy`` gathers them into the SM89 atom's fragments, and the MMA
accumulates. The waits and barriers between the copies and their consumers belong
to the implementation, as in ``tests/nvidia/cuda/fixtures/tir/e4m3_gemm.py``.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

M = 16
N = 8
K = 128
BK = 32


@module(
    entry="gemm",
    target=cuda.Target("nvidia.rtx_6000_ada"),
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
                    with Mesh(("thread",), layout=(32,), names=("thread",)) as flat:
                        with flat[0:M] as rows:
                            lhs_runs = tf.reshard(
                                a[:, k],
                                ShardLayout(layout=Layout((M, BK), (K, 1)), attrs=(S(0),), mesh=rows),
                            )
                            lhs_stage = tf.empty(
                                Tensor[
                                    (M, BK), "fp8e4m3",
                                    ShardLayout(layout=Layout((M, BK), (BK, 1)), attrs=(S(0),), mesh=rows),
                                    "smem",
                                ]
                            )
                            lhs = tf.schedule((lhs_runs, lhs_stage), op=T.copy_async())
                    lhs = tf.reshard(lhs, Layout((M, BK), (BK, 1)))
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
                    with Mesh(("thread",), layout=(32,), names=("thread",)) as flat:
                        with flat[0:N] as columns:
                            rhs_runs = tf.reshard(
                                b[k, :],
                                ShardLayout(layout=Layout((BK, N), (1, K)), attrs=(S(1),), mesh=columns),
                            )
                            rhs_stage = tf.empty(
                                Tensor[
                                    (BK, N), "fp8e4m3",
                                    ShardLayout(layout=Layout((BK, N), (1, BK)), attrs=(S(1),), mesh=columns),
                                    "smem",
                                ]
                            )
                            rhs = tf.schedule((rhs_runs, rhs_stage), op=T.copy_async())
                    rhs = tf.reshard(rhs, Layout((BK, N), (1, BK)))
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
