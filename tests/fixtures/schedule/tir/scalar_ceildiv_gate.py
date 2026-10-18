# analysis target=nvidia.h200_sxm module=ScalarCeildivGate function=gemm topology=cta wave=4/4
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r128B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r8B/w32B@logical,r32B/w128B@total,r8B/w32B@cta,r8B/w32B@thread footprint=x:40B footprint-precision=exact peak=gmem:256B;rmem:32B persistent=gmem:256B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(64,), "f32"], out: Tensor[(8,), "f32", (8,), "rmem"]):
    with Mesh((Topology("cta", 4),), layout=(4,), names=("b",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as scope:
            value = T.tensor_view(T.ptr_of(x), layout=(64,), shape=(64,))
            value_1 = T.tensor_view(
                T.ptr_of(value[ceildiv(cta.b, 2):ceildiv(cta.b, 2) + 1]),
                layout=(8,),
                shape=(8,),
            )
            value_2 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            value_3 = T.tensor_view(T.ptr_of(value_2), layout=(8,), shape=(8,))
            T.copy(value_1, value_3)
