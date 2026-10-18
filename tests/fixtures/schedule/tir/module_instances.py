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
                T.fill(out, 0.0)
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                T.copy(x, value_dst)
                weight_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                T.copy(w, weight_dst)
                scaled_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                T.binary(value_dst, weight_dst, scaled_dst, kind=BinaryKind.MUL)
                T.copy(scaled_dst, out)

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
                T.fill(out, 0.0)
                value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
                T.copy(x, value_dst)
                T.copy(value_dst, out)

    # analysis target=nvidia.h200_sxm module=ModuleInstances function=gemm topology=cta wave=1/1
    # selection requested=memory executed=memory
    # memory traffic=gmem:r2.12KB/w2.00KB@logical,r2.12KB/w2.25KB@total,r2.12KB/w2.25KB@cta,r2.12KB/w2.25KB@thread;rmem:r2.43KB/w2.38KB@logical,r2.43KB/w2.38KB@total,r2.43KB/w2.38KB@cta,r2.43KB/w2.38KB@thread footprint=layers/w:384B;mm/w:128B;state:128B;v1:50:128B;v3:44:128B;v7:25:256B;x:128B footprint-precision=exact peak=gmem:1.00KB;rmem:384B persistent=gmem:768B
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], scratch_1: Tensor[(32,), "f32"], scratch_2: Tensor[(32,), "f32"], scratch_3: Tensor[(32,), "f32"], scratch_4: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
                out_1_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(x, out_1_1)
            for i in range(0, 3, 1):
                out_1_2 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                updated_1 = T.tensor_view(T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,))
                layers[i](out_1_2, updated_1)
                with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope_1:
                    updated_2 = T.tensor_view(
                        T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,)
                    )
                    out_1_3 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
                    T.copy(updated_2, out_1_3)
            out_1_4 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            value_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 32]), layout=(32,), shape=(32,))
            mm(out_1_4, value_1)
            x_1_1 = T.tensor_view(T.ptr_of(scratch_3[0:0 + 32]), layout=(32,), shape=(32,))
            mm(value_1, x_1_1)
            x_2_1 = T.tensor_view(T.ptr_of(scratch_4[0:0 + 32]), layout=(32,), shape=(32,))
            layers[0](x_1_1, x_2_1)
            helper(x_2_1, w, out)

del ScheduledScale
