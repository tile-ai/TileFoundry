# analysis target=nvidia.h200_sxm module=ScalarBinary function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r128B/w0@logical,r128B/w0@total,r128B/w0@cta,r128B/w0@thread;rmem:r264B/w384B@logical,r264B/w384B@total,r264B/w384B@cta,r264B/w384B@thread footprint=x:128B footprint-precision=exact peak=gmem:128B;rmem:128B persistent=gmem:128B

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import B, Layout, Mesh, Topology
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        value = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        shifted = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        held = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        with Mesh((Topology("thread", 32),), Layout((32,), (1,)), names=("d0",)) as scope:
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1.0)
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(held[0:0 + 32]),
                    layout=((32,), (1,), {threads.d0 @ B()}),
                    shape=(32,),
                )
                T.copy(x, dst_frame)
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 0.25)
            with scope as threads_1:
                lhs_frame = T.tensor_view(
                    T.ptr_of(held[0:0 + 32]),
                    layout=((32,), (1,), {threads_1.d0 @ B()}),
                    shape=(32,),
                )
                T.binary(lhs_frame, scalar_1, shifted, kind=BinaryKind.ADD)
            with scope[:] as threads_2:
                T.binary(scalar, shifted, value, kind=BinaryKind.SUB)
        T.copy(value, out)
