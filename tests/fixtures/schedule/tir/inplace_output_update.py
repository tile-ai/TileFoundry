from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class InplaceOutputUpdate:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def update(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"], out_1: Tensor[(2, 4), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
                value = T.alloc_tensor(tensor_type=Tensor[(1, 4), "f32", (1, 4), "rmem"])
                value_1 = T.tensor_view(T.ptr_of(x), layout=(1, 4), shape=(1, 4))
                value_2 = T.tensor_view(T.ptr_of(value), layout=(1, 4), shape=(1, 4))
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(out_1), layout=(2, 4), shape=(2, 4))
                window = T.tensor_view(
                    T.ptr_of(value_3[1:1 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_2, window)

    # analysis target=nvidia.h200_sxm module=InplaceOutputUpdate function=gemm topology=cta wave=1/1
    # selection requested=memory executed=memory
    # memory traffic=gmem:r16B/w16B@logical,r16B/w16B@total,r16B/w16B@cta,r16B/w16B@thread;rmem:r32B/w16B@logical,r32B/w16B@total,r32B/w16B@cta,r32B/w16B@thread footprint=v2:17:16B;x:16B footprint-precision=exact peak=gmem:48B;rmem:16B persistent=gmem:48B
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"], out_1: Tensor[(2, 4), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            value = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            value_1 = T.tensor_view(T.ptr_of(x), layout=(1, 4), shape=(1, 4))
            value_2 = T.tensor_view(T.ptr_of(out_1), layout=(2, 4), shape=(2, 4))
            update(value, value_1, value_2)
