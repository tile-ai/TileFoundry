# analysis target=nvidia.h200_sxm module=MeshCaptureOutside function=gemm topology=cta wave=2/2
# selection requested=memory executed=memory
# memory traffic=gmem:r256B/w256B@logical,r384B/w384B@total,r256B/w256B@cta,r256B/w256B@thread footprint=v0:13:128B;v2:16:128B;x:128B footprint-precision=exact peak=gmem:256B persistent=gmem:128B
#   buffer=x holds=384B time=none space=cta.block reuse=128B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
    with Mesh((Topology("cta", 2),), layout=(2,), names=("d0",)) as scope:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope_1:
            value_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            T.copy(x, value_1)
    with Mesh((Topology("cta", 2),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope_2:
            value_2 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            T.copy(value_2, out)
