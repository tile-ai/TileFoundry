from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ScalarBinary:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(32,), "f32"], lhs: Tensor[(1,), "f32"], out: Tensor[(32,), "f32", "rmem"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                left_dst = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", "rmem"])
                value = T.tensor_view(T.ptr_of(lhs), layout=(1,), shape=(1,))
                value_1 = T.tensor_view(T.ptr_of(left_dst), layout=(1,), shape=(1,))
                T.copy(value, value_1)
                shifted_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                held_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                value_2 = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_3 = T.tensor_view(T.ptr_of(held_dst), layout=(32,), shape=(32,))
                T.copy(value_2, value_3)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 0.25)
                value_4 = T.tensor_view(T.ptr_of(shifted_dst), layout=(32,), shape=(32,))
                T.binary(value_3, scalar, value_4, kind=BinaryKind.ADD)
                offset = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(scalar_1, value_4, offset, kind=BinaryKind.SUB)
                scaled = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 0.5)
                T.binary(offset, scalar_2, scaled, kind=BinaryKind.MUL)
                value_5 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                T.binary(value_1, scaled, value_5, kind=BinaryKind.MUL)
