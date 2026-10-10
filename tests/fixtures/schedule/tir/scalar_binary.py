# analysis target=nvidia.h200_sxm module=ScalarBinary function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r132B/w0@logical,r132B/w0@total,r132B/w0@cta,r132B/w0@thread;rmem:r528B/w644B@logical,r528B/w644B@total,r528B/w644B@cta,r528B/w644B@thread footprint=lhs:4B;x:128B footprint-precision=exact peak=gmem:132B;rmem:132B persistent=gmem:132B

from __future__ import annotations

from tilefoundry.dsl import *  # noqa: F401, F403
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32,), "f32"], lhs: Tensor[(1,), "f32"], out: Tensor[(32,), "f32"]):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        value = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        left = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", Layout((1,), (1,)), "rmem"])
        scaled = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        offset = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        shifted = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        held = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", Layout((32,), (1,)), "rmem"])
        with Mesh((Topology("thread", 32),), Layout((32,), (1,)), names=("d0",)) as scope:
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(left[0:0 + 1]), layout=((1,), (1,), {threads.d0 @ B()}), shape=(1,)
                )
                T.copy(lhs, dst_frame)
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1.0)
            with scope as threads_1:
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(held[0:0 + 32]),
                    layout=((32,), (1,), {threads_1.d0 @ B()}),
                    shape=(32,),
                )
                T.copy(x, dst_frame_1)
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 0.25)
            with scope as threads_2:
                lhs_frame = T.tensor_view(
                    T.ptr_of(held[0:0 + 32]),
                    layout=((32,), (1,), {threads_2.d0 @ B()}),
                    shape=(32,),
                )
                T.binary(lhs_frame, scalar_1, shifted, kind=BinaryKind.ADD)
            with scope[:] as threads_3:
                T.binary(scalar, shifted, offset, kind=BinaryKind.SUB)
            scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_2, 0.5)
            with scope[:] as threads_4:
                T.binary(offset, scalar_2, scaled, kind=BinaryKind.MUL)
            with scope as threads_5:
                lhs_frame_1 = T.tensor_view(
                    T.ptr_of(left[0:0 + 1]),
                    layout=((1,), (1,), {threads_5.d0 @ B()}),
                    shape=(1,),
                )
                T.binary(lhs_frame_1, scaled, value, kind=BinaryKind.MUL)
        T.copy(value, out)
