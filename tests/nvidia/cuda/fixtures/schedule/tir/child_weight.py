from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="run")
class Child:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def run(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], out: Tensor[(32,), "f32", "rmem"]
    ):
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as cta:
            value = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
            value_1 = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
            value_2 = T.tensor_view(T.ptr_of(value), layout=(32,), shape=(32,))
            T.copy(value_1, value_2)
            value_3 = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
            value_4 = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
            value_5 = T.tensor_view(T.ptr_of(value_3), layout=(32,), shape=(32,))
            T.copy(value_4, value_5)
            value_6 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
            T.binary(value_2, value_5, value_6, kind=BinaryKind.MUL)

@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ChildWeight:
    scaled = Child

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32", "rmem"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                value = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                scaled(value, value_1)

del Child
