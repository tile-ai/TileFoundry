from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="run",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ScheduledScale:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def run(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                T.fill(out, 0.0)
                scaled_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                value = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(value_dst), layout=(32,), shape=(32,))
                T.copy(value, value_1)
                weight_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                value_2 = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
                value_3 = T.tensor_view(T.ptr_of(weight_dst), layout=(32,), shape=(32,))
                T.copy(value_2, value_3)
                value_4 = T.tensor_view(T.ptr_of(scaled_dst), layout=(32,), shape=(32,))
                T.binary(value_1, value_3, value_4, kind=BinaryKind.MUL)
                value_5 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                window = T.tensor_view(T.ptr_of(value_5[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(value_4, window)

@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ModuleInstances:
    layers = ModuleList(ScheduledScale, 3)
    mm = ScheduledScale

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def helper(x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                T.fill(out, 0.0)
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", "rmem"])
                value = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(value_dst), layout=(32,), shape=(32,))
                T.copy(value, value_1)
                value_2 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                window = T.tensor_view(T.ptr_of(value_2[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(value_1, window)

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], scratch_1: Tensor[(32,), "f32"], scratch_2: Tensor[(32,), "f32"], scratch_3: Tensor[(32,), "f32"], scratch_4: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                value = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                out_1_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(out_1_1), layout=(32,), shape=(32,))
                T.copy(value, value_1)
            for i in range(0, 3, 1):
                out_1_2 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                value_2 = T.tensor_view(T.ptr_of(out_1_2), layout=(32,), shape=(32,))
                updated_1 = T.tensor_view(T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,))
                layers[i](value_2, updated_1)
                with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope_1:
                    updated_2 = T.tensor_view(
                        T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,)
                    )
                    value_3 = T.tensor_view(T.ptr_of(updated_2), layout=(32,), shape=(32,))
                    out_1_3 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                    value_4 = T.tensor_view(T.ptr_of(out_1_3), layout=(32,), shape=(32,))
                    T.copy(value_3, value_4)
            out_1_4 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            value_6 = T.tensor_view(T.ptr_of(out_1_4), layout=(32,), shape=(32,))
            value_5_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 32]), layout=(32,), shape=(32,))
            mm(value_6, value_5_1)
            x_1_1 = T.tensor_view(T.ptr_of(scratch_3[0:0 + 32]), layout=(32,), shape=(32,))
            mm(value_5_1, x_1_1)
            x_2_1 = T.tensor_view(T.ptr_of(scratch_4[0:0 + 32]), layout=(32,), shape=(32,))
            layers[0](x_1_1, x_2_1)
            value_7 = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
            value_8 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
            helper(x_2_1, value_7, value_8)

del ScheduledScale
