# analysis target=nvidia.h200_sxm module=PaddedStages function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r24B/w72B@logical,r48B/w72B@total,r48B/w72B@cta,r48B/w72B@thread;smem:r48B/w24B@logical,r48B/w48B@total,r48B/w48B@cta,r48B/w48B@thread footprint=out:24B;x:24B footprint-precision=exact peak=gmem:56B;smem:96B persistent=gmem:24B
#   buffer=x holds=48B time=k space=none reuse=24B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(2, 3), "f32"], out: Tensor[(2, 3), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("lane",)) as scope:
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(0, dtype="f32", storage="smem", layout=((2, 3), (8, 1)), shape=(2, 3)),
                T.tensor_view(48, dtype="f32", storage="smem", layout=((2, 3), (8, 1)), shape=(2, 3)),
            )
            for k in range(0, 2, 1):
                T.copy(x, value_stages[k % 2])
                T.copy(value_stages[k % 2], out)
