"""Retain the inner accumulator tie when its initializer adopts an output window."""

from tilefoundry import func, module
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class LoopCarryWindow:
    @func
    def gemm(x: Tensor[(4, 4), "f32"]) -> Tensor[(4, 4), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                out = tf.zeros(Tensor[(4, 4), "f32", "rmem"])
                for m in range(2):
                    acc = tf.zeros(Tensor[(2, 4), "f32", "rmem"])
                    for k in range(3):
                        part = tf.schedule(
                            (x[2 * m : 2 * m + 2, :], tf.empty(Tensor[(2, 4), "f32", "rmem"])),
                            op=T.copy(),
                        )
                        acc = tf.schedule((acc, part, acc), op=T.binary(kind=BinaryKind.ADD))
                    out = tf.insert_slice(out, acc, (2 * m, 0))
                output = tf.zeros(Tensor[(4, 4), "f32"])
                return tf.insert_slice(output, out, (0, 0))
