# analysis target=nvidia.h200_sxm module=LoopCarryFromParam function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16B/w16B@logical,r16B/w16B@total,r16B/w16B@cta,r16B/w16B@thread;rmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread footprint=v0:24:8B;y:8B footprint-precision=exact peak=gmem:48B;rmem:0 persistent=gmem:24B
#   buffer=y holds=16B time=r space=none reuse=8B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(2, 4), "bf16"], y: Tensor[(1, 4), "bf16"], scratch: Tensor[(8,), "bf16"], out: Tensor[(2, 4), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        acc_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 8]), layout=(2, 4), shape=(2, 4))
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.copy(x, acc_1)
            for r in range(0, 2, 1):
                window = T.tensor_view(
                    T.ptr_of(acc_1[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(y, window)
        T.copy(acc_1, out)
