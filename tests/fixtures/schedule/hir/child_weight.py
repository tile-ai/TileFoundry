"""Inline a child whose captured weight remains a ConstTensor parameter.

Only the parent is a file-level Module; it retains the child in its slot.
"""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="run")
class Child:
    @func
    def run(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"]) -> Tensor[(32,), "f32", "rmem"]:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as _threads:
            held = tf.schedule((x, tf.empty(Tensor[(32,), "f32", (32,), "rmem"])), op=T.copy())
            wr = tf.schedule((w, tf.empty(Tensor[(32,), "f32", (32,), "rmem"])), op=T.copy())
            result = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
            return tf.schedule((held, wr, result), op=T.binary(kind=BinaryKind.MUL))


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)))
class ChildWeight:
    scaled = Child

    @func
    def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _a:
            with Mesh(("thread",), layout=(32,), names=("l",)) as _t:
                return scaled(x)  # noqa: F821


del Child
