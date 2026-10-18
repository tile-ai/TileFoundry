# analysis target=nvidia.h200_sxm module=MeshZerosSubmesh function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r0/w128B@logical,r0/w128B@total,r0/w128B@cta,r0/w128B@thread;smem:r128B/w128B@logical,r128B/w128B@total,r128B/w128B@cta,r128B/w128B@thread footprint=v1:15:128B footprint-precision=exact peak=gmem:256B;smem:128B persistent=gmem:128B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
            with scope[:16] as scope_1:
                tile = T.tensor_view(0, dtype="f32", storage="smem", layout=(32,), shape=(32,))
                T.fill(tile, 0.0)
            tile_2 = T.tensor_view(0, dtype="f32", storage="smem", layout=(32,), shape=(32,))
            T.copy(tile_2, out)
