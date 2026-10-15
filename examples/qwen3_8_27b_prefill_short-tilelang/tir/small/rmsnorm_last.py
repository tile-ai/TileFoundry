# analysis target=nvidia.h200_sxm module=SCHED_RMSNORM_LAST function=rmsnorm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r30.00KB/w30.00KB@logical,r30.00KB/w30.00KB@total,r30.00KB/w30.00KB@cta,r10.62KB/w20.31KB@thread;rmem:r190.07KB/w150.02KB@logical,r190.07KB/w150.02KB@total,r190.07KB/w150.02KB@cta,r4.52KB/w3.52KB@thread footprint=gamma:10.00KB;v22:34:10.00KB;x:10.00KB footprint-precision=lower_bound peak=gmem:50.00KB;rmem:40.01KB persistent=gmem:20.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def rmsnorm(
    x: Tensor[(1, 5120), "bf16"], gamma: Tensor[(5120,), "bf16"], scratch: Tensor[(5120,), "bf16"], out: Tensor[(1, 5120), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 5120]), layout=(1, 5120), shape=(1, 5120)
        )
        y = T.alloc_tensor(tensor_type=Tensor[(1, 5120), "bf16", (1, 32, 160), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(1, 5120), "f32", (1, 32, 160), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(1, 5120), "f32", (1, 32, 160), "rmem"])
        xf = T.alloc_tensor(tensor_type=Tensor[(1, 5120), "f32", (1, 32, 160), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "bf16", ((1, 40, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
        value_6 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
        ms = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", ((1, 1, 1), (0, 0, 0)), "rmem"])
        value_7 = T.alloc_tensor(tensor_type=Tensor[(1, 5120), "f32", (1, 32, 160), "rmem"])
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "f32", ((1, 40, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "f32", ((1, 40, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "bf16", ((1, 40, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        with Mesh((Topology("thread", 32),), layout=(1, 32), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[cta.d0 * 1:cta.d0 * 1 + 1, 0:0 + 5120]),
                layout=(1, 5120),
                shape=(1, 5120),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 5120]),
                    layout=((1 @ threads.d0, 40, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                T.copy(tile, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 5120]),
                    layout=((1 @ threads.d0, 40, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(xf[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                lhs_frame = T.tensor_view(
                    T.ptr_of(xf[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(xf[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.binary(lhs_frame, rhs_frame, dst_frame_2, kind=BinaryKind.MUL)
                src_frame_1 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(ms[0:0 + 1, 0:0 + 1]),
                    layout=((1, 1, 1), (0, 0, 0)) @ {B()},
                    shape=(1, 1),
                )
                T.reduce(
                    src_frame_1,
                    dst_frame_3,
                    axes=(-1,),
                    keepdim=True,
                    kind=ReduceKind.MEAN,
                )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            with scope as threads_4:
                lhs_frame_1 = T.tensor_view(
                    T.ptr_of(ms[0:0 + 1, 0:0 + 1]),
                    layout=((1, 1, 1), (0, 0, 0)) @ {B()},
                    shape=(1, 1),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 1]), layout=(1, 1) @ {B()}, shape=(1, 1)
                )
                T.binary(lhs_frame_1, scalar, dst_frame_4, kind=BinaryKind.ADD)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 1]), layout=(1, 1) @ {B()}, shape=(1, 1)
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 1]), layout=(1, 1) @ {B()}, shape=(1, 1)
                )
                T.unary(src_frame_2, dst_frame_5, kind=UnaryKind.RSQRT)
                lhs_frame_2 = T.tensor_view(
                    T.ptr_of(xf[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_4.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                rhs_frame_1 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 1]), layout=(1, 1) @ {B()}, shape=(1, 1)
                )
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_4.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.binary(lhs_frame_2, rhs_frame_1, dst_frame_6, kind=BinaryKind.MUL)
            tile_1 = T.tensor_view(
                T.ptr_of(gamma[0:0 + 5120]), layout=(1, 5120), shape=(1, 5120)
            )
            with scope as threads_7:
                dst_frame_7 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                T.copy(tile_1, dst_frame_7)
                src_frame_3 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_8 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                T.cast(src_frame_3, dst_frame_8, dtype="f32")
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            with scope as threads_9:
                lhs_frame_3 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_9 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                T.binary(lhs_frame_3, scalar_1, dst_frame_9, kind=BinaryKind.ADD)
                lhs_frame_4 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_9.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                rhs_frame_2 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 40, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_10 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_9.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.binary(lhs_frame_4, rhs_frame_2, dst_frame_10, kind=BinaryKind.MUL)
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_9.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                dst_frame_11 = T.tensor_view(
                    T.ptr_of(y[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_9.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.cast(src_frame_4, dst_frame_11, dtype="bf16")
                src_frame_5 = T.tensor_view(
                    T.ptr_of(y[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 32 @ threads_9.d1, 160), (5120, 160, 1)),
                    shape=(1, 5120),
                )
                T.copy(src_frame_5, value_1)
            window = T.tensor_view(
                T.ptr_of(out[cta.d0 * 1:cta.d0 * 1 + 1, 0:0 + 5120]),
                layout=(1, 5120),
                shape=(1, 5120),
            )
            T.copy(value_1, window)
