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
    def run(
        x: Tensor[(32,), "f32", (32,)], w: ConstTensor[(32,), "f32", (32,)], out: Tensor[(32,), "f32", (32,)]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            T.fill(out, 0.0)
            scaled_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            weight_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                with scope[:] as threads:
                    T.copy(x, value_dst)
                    T.copy(w, weight_dst)
                    T.binary(value_dst, weight_dst, scaled_dst, kind=BinaryKind.MUL)
                window = T.tensor_view(T.ptr_of(out[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(scaled_dst, window)

@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class ModuleInstances:
    layers = ModuleList(ScheduledScale, 3)
    mm = ScheduledScale

    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def helper(
        x: Tensor[(32,), "f32", (32,)], w: ConstTensor[(32,), "f32", (32,)], out: Tensor[(32,), "f32", (32,)]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            T.fill(out, 0.0)
            value_dst = T.alloc_tensor(tensor_type=Tensor[(32,), "f32", (32,), "rmem"])
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                with scope[:] as threads:
                    T.copy(x, value_dst)
                window = T.tensor_view(T.ptr_of(out[0:0 + 32]), layout=(32,), shape=(32,))
                T.copy(value_dst, window)

    # analysis target=nvidia.h200_sxm module=ModuleInstances function=gemm topology=cta wave=1/1
    # selection requested=memory executed=memory
    # memory traffic=gmem:r1.62KB/w1.50KB@logical,r1.62KB/w1.75KB@total,r1.62KB/w1.75KB@cta,r1.62KB/w1.75KB@thread;rmem:r2.43KB/w2.38KB@logical,r2.43KB/w2.38KB@total,r2.43KB/w2.38KB@cta,r2.43KB/w2.38KB@thread footprint=gemm/helper/v3:44:128B;gemm/layers[0]/v7:25_4:256B;gemm/layers[0]/w_4:128B;gemm/layers[i]/v0:24:128B;gemm/layers[i]/v7:25:128B;gemm/layers[i]/w:128B;gemm/mm#0/v7:25_2:256B;gemm/mm#0/w_2:128B;gemm/mm#1/v7:25_3:256B;gemm/mm#1/w_3:128B;x:128B footprint-precision=exact peak=gmem:1.12KB;rmem:128B persistent=gmem:896B
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        x: Tensor[(32,), "f32"], w: ConstTensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], scratch_1: Tensor[(32,), "f32"], scratch_2: Tensor[(32,), "f32"], scratch_3: Tensor[(32,), "f32"], scratch_4: Tensor[(32,), "f32"], scratch_5: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]
    ):
        x_1_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
        T.copy(x, x_1_1)
        for i in range(0, 3, 1):
            x_2_1 = T.tensor_view(T.ptr_of(scratch_1[0:0 + 32]), layout=(32,), shape=(32,))
            layers[i](x_1_1, x_2_1)
            T.copy(x_2_1, x_1_1)
        value_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 32]), layout=(32,), shape=(32,))
        mm(x_1_1, value_1)
        x_3_1 = T.tensor_view(T.ptr_of(scratch_3[0:0 + 32]), layout=(32,), shape=(32,))
        mm(value_1, x_3_1)
        x_4_1 = T.tensor_view(T.ptr_of(scratch_4[0:0 + 32]), layout=(32,), shape=(32,))
        layers[0](x_3_1, x_4_1)
        value_2_1 = T.tensor_view(T.ptr_of(scratch_5[0:0 + 32]), layout=(32,), shape=(32,))
        argument = T.tensor_view(T.ptr_of(w), layout=(32,), shape=(32,))
        helper(x_4_1, argument, value_2_1)
        T.copy(value_2_1, out)

del ScheduledScale
