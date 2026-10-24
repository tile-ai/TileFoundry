"""Retain indexed, repeated, constant-index, and local calls to scheduled definitions."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


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
                    value_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    value = tf.schedule((x, value_dst), op=T.copy())
                    weight_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    weight = tf.schedule((w, weight_dst), op=T.copy())
                    scaled_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    scaled = tf.schedule((value, weight, scaled_dst), op=T.binary(kind=BinaryKind.MUL))
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
                    value_dst = tf.empty(Tensor[(32,), "f32", (32,), "rmem"])
                    value = tf.schedule((x, value_dst), op=T.copy())
                    out = tf.zeros(Tensor[(32,), "f32", None, "gmem"])
                    return tf.insert_slice(out, value, (0,))

        @func
        def gemm(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"]) -> Tensor[(32,), "f32"]:
            with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
                with Mesh(("thread",), layout=(32,), names=("lane",)) as _thread:
                    out = tf.empty(Tensor[(32,), "f32", None, "gmem"])
                    state = tf.schedule((x, out), op=T.copy())
                for i in range(3):
                    updated = layers[i](state)  # noqa: F821 -- class-body binding
                    with Mesh(("thread",), layout=(32,), names=("lane",)) as _thread:
                        state = tf.schedule((updated, state), op=T.copy())
                x = mm(mm(state))  # noqa: F821 -- class-body binding
                x = layers[0](x)  # noqa: F821 -- class-body binding
                return helper(x, w)  # noqa: F821 -- class-body binding


    return ModuleInstances


ModuleInstances = _program()
