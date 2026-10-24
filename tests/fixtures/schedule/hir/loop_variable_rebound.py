"""Keep a tensor alive across a loop which reuses its authored name."""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class LoopVariableRebound:
    @func
    def gemm(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _threads:
                u = tf.zeros(Tensor[(2, 4), "f32", (2, 4), "rmem"])
                acc = tf.schedule((x, tf.empty(Tensor[(2, 4), "f32", (2, 4), "rmem"])), op=T.copy())
                for u in range(2):
                    part = tf.schedule((x, tf.empty(Tensor[(2, 4), "f32", (2, 4), "rmem"])), op=T.copy())
                    acc = tf.schedule((acc[:, :], part, acc), op=T.binary(kind=BinaryKind.ADD))
                return tf.insert_slice(u, acc, (0, 0))
