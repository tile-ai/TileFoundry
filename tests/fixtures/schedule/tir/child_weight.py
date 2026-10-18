from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="run")
class Child:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def run(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], out: Tensor[(32,), "f32", (32,), "rmem"]
    ):
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as cta:
            value = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            T.copy(x, value)
            value_1 = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            T.copy(w, value_1)
            T.binary(value, value_1, out, kind=BinaryKind.MUL)

@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ChildWeight:
    scaled = Child

    # analysis target=nvidia.h200_sxm module=ChildWeight function=gemm topology=cta wave=1/1
    # selection requested=memory executed=memory
    # memory traffic=gmem:r256B/w0@logical,r256B/w0@total,r256B/w0@cta,r256B/w0@thread;rmem:r256B/w384B@logical,r256B/w384B@total,r256B/w384B@cta,r256B/w384B@thread footprint=scaled/w:128B;x:128B footprint-precision=exact peak=gmem:256B;rmem:384B persistent=gmem:256B
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32", (32,), "rmem"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("b",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("l",)) as scope:
                scaled(x, out)

del Child
