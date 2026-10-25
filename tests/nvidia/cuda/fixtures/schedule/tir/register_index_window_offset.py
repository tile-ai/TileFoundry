from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 1),),
)
class RegisterIndexWindowOffset:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        x: Tensor[(64,), "f32"], starts: Tensor[(1,), "i64"], out: Tensor[(8,), "f32", "rmem"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
                value = T.alloc_tensor(tensor_type=Tensor[(1,), "i64", "rmem"])
                value_1 = T.tensor_view(T.ptr_of(starts), layout=(1,), shape=(1,))
                value_2 = T.tensor_view(T.ptr_of(value), layout=(1,), shape=(1,))
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(value), layout=(), shape=())
                value_4 = T.tensor_view(
                    T.ptr_of(x[value_3 + 1:value_3 + 1 + 1]), layout=(8,), shape=(8,)
                )
                value_5 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
                T.copy(value_4, value_5)
