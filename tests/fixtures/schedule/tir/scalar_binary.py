# analysis target=nvidia.h200_sxm module=ScalarBinary function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r132B/w0@logical,r132B/w0@total,r132B/w0@cta,r132B/w0@thread;rmem:r528B/w644B@logical,r528B/w644B@total,r528B/w644B@cta,r528B/w644B@thread footprint=lhs:4B;x:128B footprint-precision=exact peak=gmem:132B;rmem:132B persistent=gmem:132B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(32,), "f32"], lhs: Tensor[(1,), "f32"], out: Tensor[(32,), "f32", (32,), "rmem"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            left_dst = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
            T.copy(lhs, left_dst)
            shifted_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            held_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            T.copy(x, held_dst)
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 0.25)
            T.binary(held_dst, scalar, shifted_dst, kind=BinaryKind.ADD)
            offset = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(scalar_1, shifted_dst, offset, kind=BinaryKind.SUB)
            scaled = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_2, 0.5)
            T.binary(offset, scalar_2, scaled, kind=BinaryKind.MUL)
            T.binary(left_dst, scaled, out, kind=BinaryKind.MUL)
