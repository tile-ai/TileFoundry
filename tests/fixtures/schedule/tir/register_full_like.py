# analysis target=nvidia.h200_sxm module=RegisterFullLike function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r133B/w129B@logical,r133B/w129B@total,r133B/w129B@cta,r133B/w129B@thread footprint=x:32B footprint-precision=exact peak=gmem:32B;rmem:65B persistent=gmem:32B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "f32"], out: Tensor[(8,), "f32", (8,), "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("b",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as scope:
            value = T.alloc_tensor(tensor_type=Tensor[(8,), "f32", (8,), "rmem"])
            value_1 = T.tensor_view(T.ptr_of(x), layout=(8,), shape=(8,))
            value_2 = T.tensor_view(T.ptr_of(value), layout=(8,), shape=(8,))
            T.copy(value_1, value_2)
            value_3 = T.alloc_tensor(tensor_type=Tensor[(8,), "bool", (8,), "rmem"])
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 0.0)
            T.binary(value_2, scalar, value_3, kind=BinaryKind.GT)
            low = T.alloc_tensor(tensor_type=Tensor[(8,), "f32", (8,), "rmem"])
            T.fill(low, -1e999)
            selected = T.alloc_tensor(tensor_type=Tensor[(8,), "f32", (8,), "rmem"])
            T.where(value_3, value_2, low, selected)
            value_4 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            value_5 = T.tensor_view(T.ptr_of(value_4), layout=(8,), shape=(8,))
            T.copy(selected, value_5)
