# analysis target=nvidia.h200_sxm module=CapturedInsertUpdate function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r48B/w0@logical,r48B/w0@total,r48B/w0@cta,r48B/w0@thread;smem:r16B/w48B@logical,r16B/w48B@total,r16B/w48B@cta,r16B/w48B@thread footprint=idx:8B;x:32B footprint-precision=exact peak=gmem:48B;rmem:0;smem:16B persistent=gmem:48B
#   buffer=x holds=40B time=r space=none reuse=32B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(4, 4), "bf16"], idx: Tensor[(2,), "i64"], out: Tensor[(2, 4), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        tile = T.tensor_view(0, dtype="bf16", storage="smem", layout=(2, 4), shape=(2, 4))
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(tile, 0.0)
            for r in range(0, 2, 1):
                source = T.tensor_view(T.ptr_of(x), layout=(4, 4), shape=(4, 4))
                tile_1 = T.tensor_view(T.ptr_of(idx[r:r + 1]), layout=(1,), shape=(1,))
                with scope[:] as threads:
                    window = T.tensor_view(
                        T.ptr_of(tile[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy_async(source, window, tile_1, fill=0)
        T.copy(tile, out)
