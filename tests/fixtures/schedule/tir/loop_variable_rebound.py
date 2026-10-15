# analysis target=nvidia.h200_sxm module=LoopVariableRebound function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r152B/w160B@logical,r152B/w160B@total,r152B/w160B@cta,r152B/w160B@thread footprint=x:32B footprint-precision=exact peak=gmem:32B;rmem:32B persistent=gmem:32B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32", (2, 4), "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", (2, 4), "rmem"])
            T.copy(x, value)
            for u in range(0, 2, 1):
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                tile = T.tensor_view(
                    T.ptr_of(value[0:0 + 2, 0:0 + 4]), layout=(2, 4), shape=(2, 4)
                )
                T.binary(tile, scalar, value, kind=BinaryKind.ADD)
            window = T.tensor_view(T.ptr_of(out[0:0 + 2, 0:0 + 4]), layout=(2, 4), shape=(2, 4))
            T.copy(value, window)
