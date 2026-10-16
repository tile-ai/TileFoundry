# analysis target=nvidia.h200_sxm module=LoopOutputs function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r96B/w128B@logical,r96B/w144B@total,r96B/w144B@cta,r96B/w144B@thread;rmem:r128B/w0@logical,r128B/w0@total,r128B/w0@cta,r128B/w0@thread;smem:r32B/w32B@logical,r32B/w32B@total,r32B/w32B@cta,r32B/w32B@thread footprint=direct:16B;scratch:16B;v11:30:24B;v13:32:16B;v4:23:8B;v5:24:16B;v7:27:16B;x:16B footprint-precision=exact peak=gmem:64B;rmem:0;smem:8B persistent=gmem:16B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"), output_count=3)
def gemm(
    x: Tensor[(2, 4), "bf16"], scratch: Tensor[(8,), "bf16"], out: Tensor[(2, 4), "bf16"], out_1: Tensor[(2, 4), "bf16"], out_2: Tensor[(2, 4), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            for row in range(0, 2, 1):
                tile = T.tensor_view(
                    T.ptr_of(x[row:row + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                landing = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                T.copy_async(tile, landing)
                window = T.tensor_view(
                    T.ptr_of(out[row:row + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(landing, window)
            T.copy(x, out_1)
            for row_2 in range(0, 2, 1):
                temporary_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 8]), layout=(2, 4), shape=(2, 4)
                )
                T.fill(temporary_1, 0.0)
                tile_1 = T.tensor_view(
                    T.ptr_of(x[row_2:row_2 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                landing_1 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                T.copy_async(tile_1, landing_1)
                window_1 = T.tensor_view(
                    T.ptr_of(temporary_1[row_2:row_2 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(landing_1, window_1)
                T.copy(temporary_1, out_1)
            T.copy(out, out_2)
