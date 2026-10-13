# analysis target=nvidia.h200_sxm module=RMSNorm function=rmsnorm topology=cta wave=132/512
# selection requested=memory executed=memory
# memory traffic=gmem:r30.00KB/w10.02MB@logical,r15.00MB/w5.01GB@total,r30.00KB/w10.02MB@cta,r120B/w10.00MB@thread;rmem:r330.12KB/w290.04KB@logical,r165.06MB/w145.02MB@total,r330.12KB/w290.04KB@cta,r1.41KB/w1.17KB@thread footprint=gamma:10.00KB;hidden:20.00KB;v20:45:20.00KB footprint-precision=exact peak=gmem:30.01MB;rmem:80.02KB persistent=gmem:10.01MB
#   buffer=gamma holds=50.00KB time=none space=cta.rb reuse=1.28MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def rmsnorm(
    hidden: Tensor[(1, 1024, 5120), "bf16"], gamma: Tensor[(5120,), "bf16"], out: Tensor[(1024, 5, 256, 4), "bf16"]
):
    with Mesh((Topology("cta", 512),), layout=(512,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        y = T.alloc_tensor(tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256, 4), "rmem"])
        value = T.alloc_tensor(
            tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256, 4), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256, 4), "rmem"]
        )
        x32 = T.alloc_tensor(tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256, 4), "rmem"])
        x = T.alloc_tensor(tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256, 4), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"])
        ms = T.alloc_tensor(tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (2, 1, 1, 1), "f32", ((2, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)), "rmem"
            ]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(2, 5, 256, 4), "f32", ((2, 5, 256, 4), (20, 4, 0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256, 4), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"]
        )
        g = T.alloc_tensor(tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256, 4), "rmem"])
        with Mesh((Topology("thread", 256),), layout=(256,), names=("d0",)) as scope:
            tile = T.tensor_view(
                T.ptr_of(hidden[0:0 + 1, 0:0 + 1024, 0:0 + 5120]),
                layout=(1024, 5, 256, 4),
                shape=(1024, 5, 256, 4),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[cta.d0 * 2:cta.d0 * 2 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(2, 5, 256, 4),
                shape=(2, 5, 256, 4),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.copy(tile_1, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                src_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.unary(src_frame_1, dst_frame_2, kind=UnaryKind.SQUARE)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=((2, 5, 256 @ threads.d0, 4), (20, 4, 0, 1)),
                    shape=(2, 5, 256, 4),
                )
                T.copy(src_frame_2, dst_frame_3)
                src_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=((2, 5, 256 @ threads.d0, 4), (20, 4, 0, 1)),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 2, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((2, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)) @ {B()},
                    shape=(2, 1, 1, 1),
                )
                T.reduce(
                    src_frame_3,
                    dst_frame_4,
                    axes=(1, 2, 3),
                    keepdim=True,
                    kind=ReduceKind.MEAN,
                )
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 2, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((2, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)) @ {B()},
                    shape=(2, 1, 1, 1),
                )
                T.copy(src_frame_4, ms)
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            with scope[:] as threads_6:
                T.binary(ms, scalar, value_3, kind=BinaryKind.ADD)
                T.unary(value_3, value_2, kind=UnaryKind.RSQRT)
            with scope as threads_8:
                lhs_frame = T.tensor_view(
                    T.ptr_of(x32[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_8.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_8.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.binary(lhs_frame, value_2, dst_frame_5, kind=BinaryKind.MUL)
            tile_2 = T.tensor_view(
                T.ptr_of(gamma[0:0 + 5120]), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            with scope as threads_9:
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(g[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_9.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(tile_2, dst_frame_6)
                src_frame_5 = T.tensor_view(
                    T.ptr_of(g[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_9.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_7 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_9.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.cast(src_frame_5, dst_frame_7, dtype="f32")
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            with scope as threads_11:
                lhs_frame_1 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_8 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.binary(lhs_frame_1, scalar_1, dst_frame_8, kind=BinaryKind.ADD)
                lhs_frame_2 = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_11.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_9 = T.tensor_view(
                    T.ptr_of(value[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_11.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.binary(lhs_frame_2, rhs_frame, dst_frame_9, kind=BinaryKind.MUL)
                src_frame_6 = T.tensor_view(
                    T.ptr_of(value[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_11.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                dst_frame_10 = T.tensor_view(
                    T.ptr_of(y[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_11.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                T.cast(src_frame_6, dst_frame_10, dtype="bf16")
                y_view = T.tensor_view(
                    T.ptr_of(y[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256 @ threads_11.d0, 4),
                    shape=(2, 5, 256, 4),
                )
                window = T.tensor_view(
                    T.ptr_of(out[cta.d0 * 2:cta.d0 * 2 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(2, 5, 256, 4),
                    shape=(2, 5, 256, 4),
                )
                T.copy(y_view, window)
