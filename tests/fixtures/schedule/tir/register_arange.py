# analysis target=nvidia.h200_sxm module=RegisterArange function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=rmem:r32B/w32B@logical,r32B/w32B@total,r32B/w32B@cta,r32B/w32B@thread footprint-precision=exact peak=gmem:32B;rmem:64B persistent=gmem:32B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "i32"], out: Tensor[(8,), "i32", (8,), "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            indices = T.alloc_tensor(tensor_type=Tensor[(8,), "i32", "rmem"])
            T.arange(indices, start=0, step=1)
            value = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            T.copy(indices, value)
