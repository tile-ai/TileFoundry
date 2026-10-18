# analysis target=nvidia.h200_sxm module=SubmeshSliceInSibling function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w16B@logical,r32B/w16B@total,r32B/w16B@cta,r32B/w16B@thread;rmem:r8B/w0@logical,r8B/w0@total,r8B/w0@cta,r8B/w0@thread;smem:r16B/w32B@logical,r16B/w32B@total,r16B/w32B@cta,r16B/w32B@thread footprint=v3:17:16B;x:32B footprint-precision=exact peak=gmem:48B;rmem:0;smem:32B persistent=gmem:32B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "f32"], out: Tensor[(4,), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("b",)) as cta:
        with Mesh((Topology("thread", 64),), layout=(64,), names=("t",)) as scope:
            with scope[:32] as scope_1:
                value_1 = T.tensor_view(T.ptr_of(x), layout=(8,), shape=(8,))
                value = T.tensor_view(0, dtype="f32", storage="smem", layout=(8,), shape=(8,))
                value_2 = T.tensor_view(T.ptr_of(value), layout=(8,), shape=(8,))
                T.copy(value_1, value_2)
            with scope[32:] as scope_2:
                value_9 = T.tensor_view(0, dtype="f32", storage="smem", layout=(8,), shape=(8,))
                value_3 = T.tensor_view(T.ptr_of(value_9), layout=(8,), shape=(8,))
                value_4 = T.tensor_view(T.ptr_of(value_3), layout=(4,), shape=(4,))
                value_5 = T.tensor_view(T.ptr_of(value_4), layout=(4,), shape=(4,))
                value_6 = T.tensor_view(T.ptr_of(value_5), layout=(4,), shape=(4,))
                value_7 = T.tensor_view(T.ptr_of(out), layout=(4,), shape=(4,))
                value_8 = T.tensor_view(T.ptr_of(value_7), layout=(4,), shape=(4,))
                T.copy(value_6, value_8)
