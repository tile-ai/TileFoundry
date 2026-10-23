"""The staged producer written as ``cp.async`` instead of a descriptor copy.

A WGMMA reads its operands out of shared memory in a descriptor arrangement and
cannot move them, so whatever fills that tile has to put the bytes down that
way. Both transfers declare their landing arrangements on ``tf.empty``; the
same word: it is the transfer's own decision either way, and the consumer only
requires that the decision was made.  Nothing about the consumer changes, so
this kernel is ``wgmma_a_k_major.py`` instruction for instruction, with
``T.copy_async`` where ``T.tma`` stood.
"""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget

M = 64
N = 32
K = 32
BK = 16
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_CP_ASYNC_LOADS:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((4 @ _compute.warp, 2, 8 @ _compute.lane8, 4, 4 @ _compute.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(
                            Tensor[
                                (64, 16), "bf16",
                                Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        lhs = tf.schedule(
                            (a[:, k], lhs_ring),
                            op=T.copy_async(),
                        )
                        rhs_ring = tf.empty(
                            Tensor[
                                (16, 32), "bf16",
                                Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :], rhs_ring),
                            op=T.copy_async(),
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                output = tf.zeros(Tensor[(M, N), "bf16"])
                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                    result = tf.insert_slice(output, result, (0, 0))
                return result
