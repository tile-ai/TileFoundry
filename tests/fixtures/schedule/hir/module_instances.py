"""Retain indexed, repeated, constant-index, and local calls to scheduled definitions."""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


def _program():
    @module(
        entry="run",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)),
    )
    class ScheduledScale:
        @func
        def run(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as _thread:
                    value = tf.schedule((x,), op=T.copy(rmem_layout=(32,)))
                    weight = tf.schedule((w,), op=T.copy(rmem_layout=(32,)))
                    scaled = tf.schedule((value, weight), op=T.binary(kind=BinaryKind.MUL))
                    out = tf.zeros(Tensor[(32,), "f32", None, "gmem"])
                    return tf.insert_slice(out, scaled, (0,))

    @module(
        entry="gemm",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)),
    )
    class ModuleInstances:
        layers = ModuleList(ScheduledScale, 3)
        mm = ScheduledScale

        @func
        def helper(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            """A local call passes even an unused constant parameter explicitly."""
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as _thread:
                    value = tf.schedule((x,), op=T.copy(rmem_layout=(32,)))
                    out = tf.zeros(Tensor[(32,), "f32", None, "gmem"])
                    return tf.insert_slice(out, value, (0,))

        @func
        def gemm(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            for i in range(3):
                x = layers[i](x)  # noqa: F821 -- class-body binding
            x = mm(mm(x))  # noqa: F821 -- class-body binding
            x = layers[0](x)  # noqa: F821 -- class-body binding
            return helper(x, w)  # noqa: F821 -- class-body binding

    return ModuleInstances


ModuleInstances = _program()
