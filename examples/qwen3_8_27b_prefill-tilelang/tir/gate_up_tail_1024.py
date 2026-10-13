# analysis target=nvidia.h200_sxm module=GateUpTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r128.00KB/w8.00KB@logical,r16.00MB/w1.00MB@total,r128.00KB/w8.00KB@cta,r512B/w32B@thread;rmem:r352.35KB/w336.09KB@logical,r44.04MB/w42.01MB@total,r352.35KB/w336.09KB@cta,r1.72KB/w1.41KB@thread footprint=v41:366:8.00KB footprint-precision=lower_bound peak=gmem:84.00MB;rmem:32.02KB persistent=gmem:50.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(1024, 17408), "bf16"], ws: Tensor[(16384, 256), "f32"], out_1: Tensor[(1024, 17408), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")) as cta:
        act = T.alloc_tensor(tensor_type=Tensor[(32, 128), "bf16", (32, 4, 8, 4), "rmem"])
        value = T.alloc_tensor(tensor_type=Tensor[(32, 128), "bf16", (32, 4, 8, 4), "rmem"])
        value_1 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        g32 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "bf16", (32, 4, 8, 4), "rmem"])
        acc_g = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        acc_u = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        acc_u_1 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        acc_g_1 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_4 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_5 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_6 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_7 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "f32", (32, 4, 8, 4), "rmem"])
        value_8 = T.alloc_tensor(tensor_type=Tensor[(32, 128), "bf16", (32, 4, 8, 4), "rmem"])
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:2], layout=(32, 8), names=("d0", "d1")) as scope_1:
                tile = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 32, 0:0 + 128]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                with scope_1 as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(acc_g[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads.d0, 4, 8 @ threads.d1, 4),
                        shape=(32, 128),
                    )
                    T.copy(tile, dst_frame)
                tile_1 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 32, 128:128 + 128]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                with scope_1 as threads_1:
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(acc_u[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_1.d0, 4, 8 @ threads_1.d1, 4),
                        shape=(32, 128),
                    )
                    T.copy(tile_1, dst_frame_1)
                for j in range(1, 4, 1):
                    tile_2 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128):(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128) + 32, 128:128 + 128]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    with scope_1 as threads_2:
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_2.d0, 4, 8 @ threads_2.d1, 4),
                            shape=(32, 128),
                        )
                        T.copy(tile_2, dst_frame_2)
                        lhs_frame = T.tensor_view(
                            T.ptr_of(acc_u[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_2.d0, 4, 8 @ threads_2.d1, 4),
                            shape=(32, 128),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_2.d0, 4, 8 @ threads_2.d1, 4),
                            shape=(32, 128),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(acc_u_1[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_2.d0, 4, 8 @ threads_2.d1, 4),
                            shape=(32, 128),
                        )
                        T.binary(lhs_frame, rhs_frame, dst_frame_3, kind=BinaryKind.ADD)
                    tile_3 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128):(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128) + 32, 0:0 + 128]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    with scope_1 as threads_4:
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        T.copy(tile_3, dst_frame_4)
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(acc_g[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(acc_g_1[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        T.binary(lhs_frame_1, rhs_frame_1, dst_frame_5, kind=BinaryKind.ADD)
                        acc_g_1_view = T.tensor_view(
                            T.ptr_of(acc_g_1[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        T.copy(acc_g_1_view, acc_g)
                        acc_u_1_view = T.tensor_view(
                            T.ptr_of(acc_u_1[0:0 + 32, 0:0 + 128]),
                            layout=(32 @ threads_4.d0, 4, 8 @ threads_4.d1, 4),
                            shape=(32, 128),
                        )
                        T.copy(acc_u_1_view, acc_u)
                with scope_1 as threads_8:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc_g[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_8.d0, 4, 8 @ threads_8.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_8.d0, 4, 8 @ threads_8.d1, 4),
                        shape=(32, 128),
                    )
                    T.cast(src_frame, dst_frame_6, dtype="bf16")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_8.d0, 4, 8 @ threads_8.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(g32[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_8.d0, 4, 8 @ threads_8.d1, 4),
                        shape=(32, 128),
                    )
                    T.cast(src_frame_1, dst_frame_7, dtype="f32")
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                with scope_1 as threads_10:
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(g32[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.unary(src_frame_2, dst_frame_8, kind=UnaryKind.NEG)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.unary(src_frame_3, dst_frame_9, kind=UnaryKind.EXP)
                    rhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.binary(scalar, rhs_frame_2, dst_frame_10, kind=BinaryKind.ADD)
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(g32[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    rhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.binary(lhs_frame_2, rhs_frame_3, dst_frame_11, kind=BinaryKind.DIV)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_12 = T.tensor_view(
                        T.ptr_of(value[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.cast(src_frame_4, dst_frame_12, dtype="bf16")
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(acc_u[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_13 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.cast(src_frame_5, dst_frame_13, dtype="bf16")
                    lhs_frame_3 = T.tensor_view(
                        T.ptr_of(value[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    rhs_frame_4 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    dst_frame_14 = T.tensor_view(
                        T.ptr_of(act[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    T.binary(lhs_frame_3, rhs_frame_4, dst_frame_14, kind=BinaryKind.MUL)
                    act_view = T.tensor_view(
                        T.ptr_of(act[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ threads_10.d0, 4, 8 @ threads_10.d1, 4),
                        shape=(32, 128),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 32, (132 + cta.d0) * 128:(132 + cta.d0) * 128 + 128]),
                        layout=((32, 128), (17408, 1)),
                        shape=(32, 128),
                    )
                    T.copy(act_view, window)
        T.copy(out, out_1)
