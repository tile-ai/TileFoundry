"""Materialize a register Arange before copying it to its output."""
from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class RegisterArange:
    @func
    def gemm(x: Tensor[(8,), "i32"]) -> Tensor[(8,), "i32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as _cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as _thread:
                indices = tf.arange(Tensor[(8,), "i32", "rmem"])
                return tf.schedule((indices, tf.empty(Tensor[(8,), "i32", (8,), "rmem"])), op=T.copy())
