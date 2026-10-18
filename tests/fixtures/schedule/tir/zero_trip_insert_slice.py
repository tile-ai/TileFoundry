# analysis target=nvidia.h200_sxm module=ZeroTripInsertSlice function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16B/w48B@logical,r16B/w48B@total,r16B/w48B@cta,r16B/w48B@thread;rmem:r32B/w16B@logical,r32B/w16B@total,r32B/w16B@cta,r32B/w16B@thread footprint=v3:15:0;x:0 footprint-precision=lower_bound peak=gmem:48B;rmem:16B persistent=gmem:16B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(1, 4), "f32"], out: Tensor[(2, 4), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("b",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            T.fill(value, 0.0)
            for _i in range(0, 0, 1):
                value_1 = T.alloc_tensor(tensor_type=Tensor[(1, 4), "f32", (1, 4), "rmem"])
                value_2 = T.tensor_view(T.ptr_of(x), layout=(1, 4), shape=(1, 4))
                value_3 = T.tensor_view(T.ptr_of(value_1), layout=(1, 4), shape=(1, 4))
                T.copy(value_2, value_3)
                value_4 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                value_5 = T.tensor_view(T.ptr_of(value_4), layout=(2, 4), shape=(2, 4))
                value_6 = T.tensor_view(T.ptr_of(value_5), layout=(2, 4), shape=(2, 4))
                window = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_3, window)
