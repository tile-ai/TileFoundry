from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 4), Topology("thread", 1),),
)
class ScalarCeildivGate:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(64,), "f32"], out: Tensor[(8,), "f32", "rmem"]):
        with Mesh((Topology("cta", 4),), layout=(4,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
                value = T.tensor_view(
                    T.ptr_of(x[ceildiv(cta.d0, 2):ceildiv(cta.d0, 2) + 1]), layout=(8,), shape=(8,)
                )
                value_1 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
                T.copy(value, value_1)
