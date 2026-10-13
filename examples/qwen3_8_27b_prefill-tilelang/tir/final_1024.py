# analysis target=nvidia.h200_sxm module=Final function=final topology=cta wave=132/7760
# selection requested=memory executed=memory
# memory traffic=gmem:r340.01KB/w485.06KB@logical,r2.52GB/w3.59GB@total,r340.01KB/w485.06KB@cta,r1.34KB/w485.01KB@thread;rmem:r2.40MB/w2.38MB@logical,r18.21GB/w18.06GB@total,r2.40MB/w2.38MB@cta,r10.63KB/w10.42KB@thread;smem:r10.00KB/w10.00KB@logical,r75.78MB/w75.78MB@total,r10.00KB/w10.00KB@cta,r320B/w40B@thread footprint=<value 0>[0]:32B;gamma_final:10.00KB;hidden:10.00KB footprint-precision=unknown peak=gmem:2.38GB;rmem:960.01KB;smem:10.00KB persistent=gmem:2.38GB
#   buffer=hidden holds=20.03KB time=none space=cta.b reuse=1.28MB fits=yes precision=unknown
#   buffer=gamma_final holds=20.03KB time=none space=cta.b reuse=1.28MB fits=yes precision=unknown
#   buffer=<value 0>[0] holds=20.03KB time=none space=cta.b reuse=4.09KB fits=yes precision=unknown
#   error="rmem placement peak 960.01KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def final(
    hidden: Tensor[(1, 1024, 5120), "bf16"], last: Tensor[(1,), "i64"], gamma_final: Tensor[(5120,), "bf16"], w_head: Tensor[(248320, 5120), "bf16"], out: Tensor[(62080, 4, 1, 1, 1), "bf16"]
):
    with Mesh((Topology("cta", 7760),), layout=(7760,), names=("d0",)) as cta:
        xs = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
        )
        xn = T.alloc_tensor(tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256, 4), "rmem"])
        value = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"]
        )
        x32 = T.alloc_tensor(tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"])
        x = T.alloc_tensor(tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256, 4), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"])
        ms = T.alloc_tensor(tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (1, 1, 1, 1), "f32", ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)), "rmem"
            ]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", ((1, 5, 256, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256, 4), "rmem"]
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
            tile_1 = T.tensor_view(T.ptr_of(last[0:0 + 1]), layout=(1,), shape=(1,))
            tile_2 = T.tensor_view(T.ptr_of(tile_1[0:0 + 1]), layout=(), shape=())
            tile_3 = T.tensor_view(
                T.ptr_of(tile[tile_2:tile_2 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256, 4),
                shape=(1, 5, 256, 4),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(tile_3, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(x[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                src_frame_1 = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.unary(src_frame_1, dst_frame_2, kind=UnaryKind.SQUARE)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=((1, 5, 256 @ threads.d0, 4), (0, 4, 0, 1)),
                    shape=(1, 5, 256, 4),
                )
                T.copy(src_frame_2, dst_frame_3)
                src_frame_3 = T.tensor_view(
                    T.ptr_of(sq[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=((1, 5, 256 @ threads.d0, 4), (0, 4, 0, 1)),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)) @ {B()},
                    shape=(1, 1, 1, 1),
                )
                T.reduce(
                    src_frame_3,
                    dst_frame_4,
                    axes=(1, 2, 3),
                    keepdim=True,
                    kind=ReduceKind.MEAN,
                )
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)) @ {B()},
                    shape=(1, 1, 1, 1),
                )
                T.copy(src_frame_4, ms)
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            with scope[:] as threads_6:
                T.binary(ms, scalar, value_3, kind=BinaryKind.ADD)
                T.unary(value_3, value_2, kind=UnaryKind.RSQRT)
            with scope as threads_8:
                lhs_frame = T.tensor_view(
                    T.ptr_of(x32[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_8.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_8.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.binary(lhs_frame, value_2, dst_frame_5, kind=BinaryKind.MUL)
            tile_4 = T.tensor_view(
                T.ptr_of(gamma_final[0:0 + 5120]), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            with scope as threads_9:
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(g[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_9.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(tile_4, dst_frame_6)
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
                    T.ptr_of(value_1[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_9 = T.tensor_view(
                    T.ptr_of(value[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.binary(lhs_frame_2, rhs_frame, dst_frame_9, kind=BinaryKind.MUL)
                src_frame_6 = T.tensor_view(
                    T.ptr_of(value[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                dst_frame_10 = T.tensor_view(
                    T.ptr_of(xn[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.cast(src_frame_6, dst_frame_10, dtype="bf16")
                src_frame_7 = T.tensor_view(
                    T.ptr_of(xn[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ threads_11.d0, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(src_frame_7, xs)
        T.fill(out, 0.0)
        logit = T.alloc_tensor(
            tensor_type=Tensor[(8, 4, 1, 1, 1), "bf16", (8, 4, 1, 1, 1), "rmem"]
        )
        acc = T.alloc_tensor(
            tensor_type=Tensor[(8, 4, 1, 1, 1), "f32", (8, 4, 1, 1, 1), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 4, 1, 1, 1), "f32", ((8, 4, 1, 1, 1, 1, 1), (4, 1, 0, 0, 0, 0, 0)), "rmem"
            ]
        )
        prod = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 4, 20, 32, 8), "f32", ((8, 4, 20, 32, 8), (0, 160, 8, 0, 1)), "rmem"
            ]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(8, 4, 20, 32, 8), "f32", (8, 4, 20, 32, 8), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(8, 4, 20, 32, 8), "f32", (8, 4, 20, 32, 8), "rmem"]
        )
        wr = T.alloc_tensor(
            tensor_type=Tensor[(8, 4, 20, 32, 8), "bf16", (8, 4, 20, 32, 8), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 20, 32, 8), "f32", (1, 1, 20, 32, 8), "rmem"]
        )
        xv = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 20, 32, 8), "bf16", (1, 1, 20, 32, 8), "rmem"]
        )
        with Mesh((Topology("thread", 256),), layout=(8, 32), names=("d0", "d1")) as scope_1:
            tile_5 = T.tensor_view(
                T.ptr_of(w_head[0:0 + 248320, 0:0 + 5120]),
                layout=(62080, 4, 20, 32, 8),
                shape=(62080, 4, 20, 32, 8),
            )
            tile_6 = T.tensor_view(
                T.ptr_of(tile_5[(cta.d0 * 32) // 4:(cta.d0 * 32) // 4 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                layout=(8, 4, 20, 32, 8),
                shape=(8, 4, 20, 32, 8),
            )
            with scope_1 as threads_15:
                dst_frame_11 = T.tensor_view(
                    T.ptr_of(wr[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_15.d0, 4, 20, 32 @ threads_15.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                T.copy(tile_6, dst_frame_11)
                src_frame_8 = T.tensor_view(
                    T.ptr_of(wr[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_15.d0, 4, 20, 32 @ threads_15.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                dst_frame_12 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_15.d0, 4, 20, 32 @ threads_15.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                T.cast(src_frame_8, dst_frame_12, dtype="f32")
            tile_7 = T.tensor_view(
                T.ptr_of(xs[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 1, 20, 32, 8),
                shape=(1, 1, 20, 32, 8),
            )
            with scope_1 as threads_17:
                dst_frame_13 = T.tensor_view(
                    T.ptr_of(xv[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(1, 1, 20, 32 @ threads_17.d1, 8),
                    shape=(1, 1, 20, 32, 8),
                )
                T.copy(tile_7, dst_frame_13)
                src_frame_9 = T.tensor_view(
                    T.ptr_of(xv[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(1, 1, 20, 32 @ threads_17.d1, 8),
                    shape=(1, 1, 20, 32, 8),
                )
                dst_frame_14 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(1, 1, 20, 32 @ threads_17.d1, 8),
                    shape=(1, 1, 20, 32, 8),
                )
                T.cast(src_frame_9, dst_frame_14, dtype="f32")
                lhs_frame_3 = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_17.d0, 4, 20, 32 @ threads_17.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                rhs_frame_1 = T.tensor_view(
                    T.ptr_of(value_11[0:0 + 1, 0:0 + 1, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(1, 1, 20, 32 @ threads_17.d1, 8),
                    shape=(1, 1, 20, 32, 8),
                )
                dst_frame_15 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_17.d0, 4, 20, 32 @ threads_17.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                T.binary(lhs_frame_3, rhs_frame_1, dst_frame_15, kind=BinaryKind.MUL)
                src_frame_10 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=(8 @ threads_17.d0, 4, 20, 32 @ threads_17.d1, 8),
                    shape=(8, 4, 20, 32, 8),
                )
                dst_frame_16 = T.tensor_view(
                    T.ptr_of(prod[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=((8 @ threads_17.d0, 4, 20, 32 @ threads_17.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(8, 4, 20, 32, 8),
                )
                T.copy(src_frame_10, dst_frame_16)
                src_frame_11 = T.tensor_view(
                    T.ptr_of(prod[0:0 + 8, 0:0 + 4, 0:0 + 20, 0:0 + 32, 0:0 + 8]),
                    layout=((8 @ threads_17.d0, 4, 20, 32 @ threads_17.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(8, 4, 20, 32, 8),
                )
                dst_frame_17 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((8 @ threads_17.d0, 4, 1, 1, 1, 1, 1), (4, 1, 0, 0, 0, 0, 0)),
                    shape=(8, 4, 1, 1, 1),
                )
                T.reduce(
                    src_frame_11,
                    dst_frame_17,
                    axes=(2, 3, 4),
                    keepdim=True,
                    kind=ReduceKind.SUM,
                )
                src_frame_12 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((8 @ threads_17.d0, 4, 1, 1, 1, 1, 1), (4, 1, 0, 0, 0, 0, 0)),
                    shape=(8, 4, 1, 1, 1),
                )
                dst_frame_18 = T.tensor_view(
                    T.ptr_of(acc[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8 @ threads_17.d0, 4, 1, 1, 1),
                    shape=(8, 4, 1, 1, 1),
                )
                T.copy(src_frame_12, dst_frame_18)
                src_frame_13 = T.tensor_view(
                    T.ptr_of(acc[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8 @ threads_17.d0, 4, 1, 1, 1),
                    shape=(8, 4, 1, 1, 1),
                )
                dst_frame_19 = T.tensor_view(
                    T.ptr_of(logit[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8 @ threads_17.d0, 4, 1, 1, 1),
                    shape=(8, 4, 1, 1, 1),
                )
                T.cast(src_frame_13, dst_frame_19, dtype="bf16")
                logit_view = T.tensor_view(
                    T.ptr_of(logit[0:0 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8 @ threads_17.d0, 4, 1, 1, 1),
                    shape=(8, 4, 1, 1, 1),
                )
                window = T.tensor_view(
                    T.ptr_of(out[(cta.d0 * 32) // 4:(cta.d0 * 32) // 4 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8, 4, 1, 1, 1),
                    shape=(8, 4, 1, 1, 1),
                )
                T.copy(logit_view, window)
