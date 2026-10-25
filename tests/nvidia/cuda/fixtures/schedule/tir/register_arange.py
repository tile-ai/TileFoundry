from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 1),),
)
class RegisterArange:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(8,), "i32"], out: Tensor[(8,), "i32", "rmem"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
                indices = T.alloc_tensor(tensor_type=Tensor[(8,), "i32", "rmem"])
                T.arange(indices, start=0, step=1)
                value = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
                T.copy(indices, value)
