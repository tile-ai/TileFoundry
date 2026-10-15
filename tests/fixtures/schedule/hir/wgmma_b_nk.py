"""Schedule an NK weight through an explicit K,N shared-memory destination."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WGMMA_B_NK:
    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"],
        weight: Tensor[(16, 16), "bf16"],
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as threads:
                output = tf.zeros(Tensor[(64, 16), "f32"])
                acc = tf.zeros(Tensor[
                    (64, 16), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 2, 4 @ threads.lane4, 2,),
                     (16, 8, 1, 512, 128, 64)), "rmem",
                ])
                lhs_dst = tf.empty(Tensor[
                    (64, 16), "bf16",
                    Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), "smem",
                ])
                lhs = tf.schedule((a, lhs_dst), op=T.copy())
                rhs_kn = tf.transpose(weight, (1, 0))
                rhs_dst = tf.empty(Tensor[
                    (16, 16), "bf16",
                    Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))), "smem",
                ])
                rhs = tf.schedule((rhs_kn, rhs_dst), op=T.copy())
                result = tf.schedule(
                    (acc, lhs, rhs),
                    op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(
                        n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                        a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                    )),
                )
                return tf.insert_slice(output, result, (0, 0))
