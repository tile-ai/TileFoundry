# analysis target=nvidia.h200_sxm module=ViewVisitMemo function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r40B/w0@logical,r40B/w0@total,r40B/w0@cta,r40B/w0@thread;rmem:r16B/w32B@logical,r16B/w32B@total,r16B/w32B@cta,r16B/w32B@thread footprint=<value 0>[0]:8B;x:32B footprint-precision=upper_bound peak=gmem:520B;rmem:32B persistent=gmem:520B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(16, 8), "f32"], index: Tensor[(1,), "i64"], out: Tensor[(1, 8), "f32", "rmem"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("lane",)) as scope:
            value = T.tensor_view(T.ptr_of(index), layout=(1,), shape=(1,))
            value_1 = T.tensor_view(T.ptr_of(value), layout=(), shape=())
            value_2 = T.tensor_view(T.ptr_of(x), layout=(16, 8), shape=(16, 8))
            value_3 = T.tensor_view(
                T.ptr_of(value_2[value_1:value_1 + 1, 0:0 + 1]),
                layout=((1, 8), (0, 1)),
                shape=(1, 8),
            )
            T.copy(value_3, out)
