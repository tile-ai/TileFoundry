# analysis target=nvidia.h200_sxm module=ReshardViews function=gemm topology=cta wave=2/2
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r64B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r0/w32B@logical,r0/w64B@total,r0/w32B@cta,r0/w32B@thread footprint=x:32B footprint-precision=exact peak=gmem:32B;rmem:32B persistent=gmem:32B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "f32"], out: Tensor[(8,), "f32", (8,), "rmem"]):
    with Mesh((Topology("cta", 2),), layout=(2,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            value = T.tensor_view(T.ptr_of(x), layout=(8,), shape=(8,))
            value_1 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            T.copy(value, value_1)
