# analysis target=nvidia.h200_sxm module=TwoOutputs function=pair topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16.25KB/w0@logical,r16.25KB/w0@total,r16.25KB/w0@cta,r16.25KB/w0@thread;rmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread;smem:r0/w16.25KB@logical,r0/w16.25KB@total,r0/w16.25KB@cta,r0/w16.25KB@thread footprint=x:16.00KB footprint-precision=exact peak=gmem:16.00KB;rmem:0;smem:16.25KB persistent=gmem:16.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"), output_count=2)
def pair(
    x: Tensor[(64, 64), "f32"], out: Tensor[(64, 64), "f32"], out_1: Tensor[(64, 1), "f32"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        tile_dst = T.tensor_view(
            0, dtype="f32", storage="smem", layout=(64, 64), shape=(64, 64)
        )
        column_dst = T.tensor_view(
            16384, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
        )
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 64, 0:0 + 64]), layout=(64, 64), shape=(64, 64)
            )
            with scope[:] as threads:
                T.copy(tile, tile_dst)
            tile_1 = T.tensor_view(
                T.ptr_of(x[0:0 + 64, 0:0 + 1]), layout=((64, 1), (64, 1)), shape=(64, 1)
            )
            with scope[:] as threads_1:
                T.copy(tile_1, column_dst)
        T.copy(tile_dst, out)
        T.copy(column_dst, out_1)
