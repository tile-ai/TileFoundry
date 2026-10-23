"""Materialize a register FullLike consumed by Where and a copy."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class RegisterFullLike:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _thread:
                held = tf.schedule((x, tf.empty(Tensor[(8,), "f32", (8,), "rmem"])), op=T.copy())
                low = tf.full_like(held, value=-1e999)
                selected = tf.where(held > 0.0, held, low)
                return tf.schedule((selected, tf.empty(Tensor[(8,), "f32", (8,), "rmem"])), op=T.copy())
