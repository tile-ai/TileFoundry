from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class MeshZerosSubmesh:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                with scope[:16] as scope_1:
                    tile = T.tensor_view(0, dtype="f32", storage="smem", layout=(32,), shape=(32,))
                    T.fill(tile, 0.0)
                tile_2 = T.tensor_view(0, dtype="f32", storage="smem", layout=(32,), shape=(32,))
                value = T.tensor_view(T.ptr_of(tile_2), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                T.copy(value, value_1)
