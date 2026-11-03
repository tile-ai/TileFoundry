"""Two BF16 WGMMAs chain, then E4M3 and BF16 change shape on the same accumulator.

The second BF16 needs no step. Each shape change waits for the group and fences again.
The last reads the BF16 tiles again, so the E4M3 tiles take bytes of their own.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WgmmaShapes:
    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"],
        b: Tensor[(16, 16), "bf16"],
        a8: Tensor[(64, 32), "fp8e4m3"],
        b8: Tensor[(32, 16), "fp8e4m3"],
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as threads:
                output = tf.zeros(Tensor[(64, 16), "f32"])
                acc = tf.zeros(Tensor[
                    (64, 16), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 2, 4 @ threads.lane4, 2,),
                     (16, 8, 1, 512, 128, 64)), "rmem",
                ])
                lhs = tf.schedule(
                    (a, tf.empty(Tensor[
                        (64, 16), "bf16", Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), "smem",
                    ])),
                    op=T.copy(),
                )
                rhs = tf.schedule(
                    (tf.transpose(b, (1, 0)), tf.empty(Tensor[
                        (16, 16), "bf16", Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))), "smem",
                    ])),
                    op=T.copy(),
                )
                bf16 = T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                )
                acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=bf16))
                acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=bf16))
                lhs8 = tf.schedule(
                    (a8, tf.empty(Tensor[
                        (64, 32), "fp8e4m3", Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), "smem",
                    ])),
                    op=T.copy(),
                )
                rhs8 = tf.schedule(
                    (b8, tf.empty(Tensor[
                        (32, 16), "fp8e4m3", Layout(((2, 16), (2, 8)), ((128, 1), (256, 16))), "smem",
                    ])),
                    op=T.copy(),
                )
                acc = tf.schedule(
                    (acc, lhs8, rhs8),
                    op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(n=16, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS)),
                )
                acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=bf16))
                return tf.insert_slice(output, acc, (0, 0))
