# analysis target=nvidia.h200_sxm module=LoopCarryFromParam function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w32B@logical,r32B/w32B@total,r32B/w32B@cta,r32B/w32B@thread;rmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread footprint=v0:22:16B;v2:25:8B;x:16B;y:8B footprint-precision=exact peak=gmem:48B;rmem:0 persistent=gmem:24B
#   buffer=y holds=16B time=r space=none reuse=8B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(2, 4), "bf16"], y: Tensor[(1, 4), "bf16"], out: Tensor[(2, 4), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
            value = T.tensor_view(T.ptr_of(x), layout=(2, 4), shape=(2, 4))
            value_1 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            value_2 = T.tensor_view(T.ptr_of(value_1), layout=(2, 4), shape=(2, 4))
            T.copy(value, value_2)
            for r in range(0, 2, 1):
                value_3 = T.tensor_view(T.ptr_of(y), layout=(1, 4), shape=(1, 4))
                value_4 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                value_5 = T.tensor_view(T.ptr_of(value_4), layout=(2, 4), shape=(2, 4))
                value_6 = T.tensor_view(T.ptr_of(value_5), layout=(2, 4), shape=(2, 4))
                value_7 = T.tensor_view(T.ptr_of(value_6), layout=(2, 4), shape=(2, 4))
                window = T.tensor_view(
                    T.ptr_of(value_7[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_3, window)
