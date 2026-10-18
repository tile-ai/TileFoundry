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
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
                value = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                T.fill(value, 0.0)
                scaled_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                value_1 = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_2 = T.tensor_view(T.ptr_of(value_dst), layout=(32,), shape=(32,))
                T.copy(value_1, value_2)
                weight_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                value_3 = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
                value_4 = T.tensor_view(T.ptr_of(weight_dst), layout=(32,), shape=(32,))
                T.copy(value_3, value_4)
                value_5 = T.tensor_view(T.ptr_of(scaled_dst), layout=(32,), shape=(32,))
                T.binary(value_2, value_4, value_5, kind=BinaryKind.MUL)
                value_6 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                window = T.tensor_view(T.ptr_of(value_6[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(value_5, window)

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
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
                value = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                T.fill(value, 0.0)
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                value_1 = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                value_2 = T.tensor_view(T.ptr_of(value_dst), layout=(32,), shape=(32,))
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
                window = T.tensor_view(T.ptr_of(value_3[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(value_2, window)

    # analysis target=nvidia.h200_sxm module=ModuleInstances function=gemm topology=cta wave=1/1
    # selection requested=memory executed=memory
    # memory traffic=gmem:r2.12KB/w2.00KB@logical,r2.12KB/w2.25KB@total,r2.12KB/w2.25KB@cta,r2.12KB/w2.25KB@thread;rmem:r2.43KB/w2.38KB@logical,r2.43KB/w2.38KB@total,r2.43KB/w2.38KB@cta,r2.43KB/w2.38KB@thread footprint=layers/w:384B;mm/w:128B;state:128B;v1:50:128B;v3:44:128B;v7:25:256B;x:128B footprint-precision=exact peak=gmem:1.00KB;rmem:384B persistent=gmem:768B
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], scratch_1: Tensor[(32,), "f32"], scratch_2: Tensor[(32,), "f32"], scratch_3: Tensor[(32,), "f32"], scratch_4: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
                value = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
                out_1_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                value_1 = T.tensor_view(T.ptr_of(out_1_1), layout=(32,), shape=(32,))
                T.copy(value, value_1)
            for i in range(0, 3, 1):
                out_1_2 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                value_2 = T.tensor_view(T.ptr_of(out_1_2), layout=(32,), shape=(32,))
                value_3 = T.tensor_view(T.ptr_of(value_2), layout=(32,), shape=(32,))
                value_4 = T.tensor_view(T.ptr_of(value_3), layout=(32,), shape=(32,))
                updated_1 = T.tensor_view(T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,))
                layers[i](value_4, updated_1)
                with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope_1:
                    updated_2 = T.tensor_view(
                        T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,)
                    )
                    value_5 = T.tensor_view(T.ptr_of(updated_2), layout=(32,), shape=(32,))
                    out_1_3 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                    value_6 = T.tensor_view(T.ptr_of(out_1_3), layout=(32,), shape=(32,))
                    value_7 = T.tensor_view(T.ptr_of(value_6), layout=(32,), shape=(32,))
                    value_8 = T.tensor_view(T.ptr_of(value_7), layout=(32,), shape=(32,))
                    value_9 = T.tensor_view(T.ptr_of(value_8), layout=(32,), shape=(32,))
                    value_10 = T.tensor_view(T.ptr_of(value_9), layout=(32,), shape=(32,))
                    T.copy(value_5, value_10)
            out_1_4 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            value_12 = T.tensor_view(T.ptr_of(out_1_4), layout=(32,), shape=(32,))
            value_13 = T.tensor_view(T.ptr_of(value_12), layout=(32,), shape=(32,))
            value_14 = T.tensor_view(T.ptr_of(value_13), layout=(32,), shape=(32,))
            value_15 = T.tensor_view(T.ptr_of(value_14), layout=(32,), shape=(32,))
            value_11_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 32]), layout=(32,), shape=(32,))
            mm(value_15, value_11_1)
            x_1_1 = T.tensor_view(T.ptr_of(scratch_3[0:0 + 32]), layout=(32,), shape=(32,))
            mm(value_11_1, x_1_1)
            x_2_1 = T.tensor_view(T.ptr_of(scratch_4[0:0 + 32]), layout=(32,), shape=(32,))
            layers[0](x_1_1, x_2_1)
            value_16 = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
            value_17 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
            helper(x_2_1, value_16, value_17)

del ScheduledScale
