# analysis target=nvidia.h200_sxm module=DynWindow function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r40B/w0@logical,r40B/w0@total,r40B/w0@cta,r40B/w0@thread;rmem:r8B/w40B@logical,r8B/w40B@total,r8B/w40B@cta,r8B/w40B@thread footprint=starts:8B;x:32B footprint-precision=upper_bound peak=gmem:264B;rmem:32B persistent=gmem:264B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(64,), "f32"], starts: Tensor[(1,), "i64"], out: Tensor[(8,), "f32", (8,), "rmem"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            value = T.alloc_tensor(tensor_type=Tensor[(1,), "i64", (1,), "rmem"])
            value_1 = T.tensor_view(T.ptr_of(starts), layout=(1,), shape=(1,))
            value_2 = T.tensor_view(T.ptr_of(value), layout=(1,), shape=(1,))
            T.copy(value_1, value_2)
            value_3 = T.tensor_view(T.ptr_of(value), layout=(), shape=())
            value_4 = T.tensor_view(T.ptr_of(x[value_3:value_3 + 1]), layout=(8,), shape=(8,))
            value_5 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            T.copy(value_4, value_5)
