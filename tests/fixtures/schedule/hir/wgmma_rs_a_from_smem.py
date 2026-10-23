"""Read RS-form FP8 WGMMA A after staging it through shared memory.

TMA first stages A like any shared operand. A participant-wise ``T.copy`` then
lands it in the 64x32 FP8 register fragment, the one arrangement accepted by
the RS atom, on the warpgroup that will multiply it: each thread holds four
bytes of K per pair of rows, as a 16-bit fragment does. B stays in shared
memory and must be K-major, so it is the (K, N) view of a row-major (N, K)
weight. This contrasts with the sibling fixture where a preceding WGMMA
accumulator supplies those registers.
"""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

M = 64
N = 32
K = 64
BK = 32
STAGES = 2


_HELD = (Split(2), Split(0), Split(4))


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_RS_A_FROM_SMEM:
    @func
    def gemm(
        a: Tensor[(M, K), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", ((K, N), (1, K))],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                register = T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.RS)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((4 @ _compute.warp, 2, 8 @ _compute.lane8, 4, 4 @ _compute.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(
                            Tensor[
                                (64, 32), "fp8e4m3",
                                Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
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
                                (32, 32), "fp8e4m3",
                                Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :], rhs_ring),
                            op=T.copy_async_tensor(),
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                        frag_dst = tf.empty(
                            Tensor[
                                (64, 32), "fp8e4m3",
                                ShardLayout(
                                    layout=Layout((4, 2, 8, 2, 4, 4,), (16, 8, 1, 1024, 256, 64)),
                                    attrs=(S(0), S(2), S(4)),
                                    mesh=_compute,
                                ),
                                "rmem",
                            ],
                        )
                        frag = tf.schedule(
                            (lhs, frag_dst),
                            op=T.copy(),
                        )
                        acc = tf.schedule(
                            (acc, frag, rhs),
                            op=T.tiled_mma(atom=register),
                        )

                output = tf.zeros(Tensor[(M, N), "bf16"])
                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                    result = tf.insert_slice(output, result, (0, 0))
                return result
