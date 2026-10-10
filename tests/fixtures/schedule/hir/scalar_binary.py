"""Lower scalar and lhs size-one broadcasts through binary instructions.

Literals stay unmaterialized in HIR; explicit and automatic instructions must
allocate and fill rank-zero register tensors. The final scheduled multiply
broadcasts its size-one lhs over the register tile. Explicit scheduling with a
constant on the left is absent: tf.schedule takes the scheduled tensor as its
first operand, so a literal there would describe a different path.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


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
                held = tf.schedule((x,), op=T.copy(rmem_layout=((32,), (1,), {_threads.lane @ B()})))
                shifted = tf.schedule((held, 0.25), op=T.binary(kind=BinaryKind.ADD))
                offset = 1.0 - shifted
                scaled = offset * 0.5
                left = tf.schedule((lhs,), op=T.copy(rmem_layout=((1,), (1,), {_threads.lane @ B()})))
                return tf.schedule((left, scaled), op=T.binary(kind=BinaryKind.MUL))
