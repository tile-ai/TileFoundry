"""Lower scalar and lhs size-one broadcasts through binary instructions.

Literals stay unmaterialized in HIR; explicit and automatic instructions must
allocate and fill rank-zero register tensors. The final scheduled multiply
broadcasts its size-one lhs over the register tile. Explicit scheduling with a
constant on the left is absent: tf.schedule takes the scheduled tensor as its
first operand, so a literal there would describe a different path.
"""

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class ScalarBinary:
    @func
    def gemm(x: Tensor[(32,), "f32"], lhs: Tensor[(1,), "f32"]) -> Tensor[(32,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                held_dst = tf.empty(
                    Tensor[
                        (32,), "f32",
                        (32,),
                        "rmem",
                    ],
                )
                held = tf.schedule((x, held_dst), op=T.copy())
                shifted_dst = tf.empty(
                    Tensor[
                        (32,), "f32",
                        (32,),
                        "rmem",
                    ],
                )
                shifted = tf.schedule((held, 0.25, shifted_dst), op=T.binary(kind=BinaryKind.ADD))
                offset = 1.0 - shifted
                scaled = offset * 0.5
                left_dst = tf.empty(
                    Tensor[
                        (1,), "f32",
                        (1,),
                        "rmem",
                    ],
                )
                left = tf.schedule((lhs, left_dst), op=T.copy())
                result_dst = tf.empty(
                    Tensor[
                        (32,), "f32",
                        (32,),
                        "rmem",
                    ],
                )
                return tf.schedule((left, scaled, result_dst), op=T.binary(kind=BinaryKind.MUL))
