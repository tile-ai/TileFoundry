# analysis target=nvidia.h200_sxm module=QKVTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r136.50KB/w16.00KB@logical,r17.06MB/w2.00MB@total,r136.50KB/w16.00KB@cta,r608B/w64B@thread;rmem:r599.35KB/w527.06KB@logical,r74.92MB/w65.88MB@total,r599.35KB/w527.06KB@cta,r3.01KB/w2.44KB@thread;smem:r20.00KB/w20.00KB@logical,r2.50MB/w2.50MB@total,r20.00KB/w20.00KB@cta,r80B/w80B@thread footprint=cos:0;gamma_k:0;sin:0;v96:883:16.00KB footprint-precision=lower_bound peak=gmem:466.00MB;rmem:112.02KB;smem:16.00KB persistent=gmem:242.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(8192, 14336), "bf16"], ws: Tensor[(16384, 256), "f32"], gamma_q: Tensor[(256,), "bf16"], gamma_k: Tensor[(256,), "bf16"], cos: Tensor[(8192, 64), "bf16"], sin: Tensor[(8192, 64), "bf16"], out_1: Tensor[(8192, 14336), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(2, 16, 4), names=("d0", "d1", "d2")) as cta:
        cs = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "bf16", (32, 8, 8, 4), "rmem"])
        acc = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"])
        acc_1 = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"])
        value = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"])
        cs_1 = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "bf16", (32, 8, 8, 4), "rmem"])
        nrm = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "bf16", (32, 8, 8, 4), "rmem"])
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"]
        )
        x32 = T.alloc_tensor(tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"])
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 1, 1), "f32", (32, 1, 1, 1), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 1, 1), "f32", (32, 1, 1, 1), "rmem"]
        )
        ms = T.alloc_tensor(tensor_type=Tensor[(32, 1, 1, 1), "f32", (32, 1, 1, 1), "rmem"])
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[
                (32, 1, 1, 1), "f32", ((32, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)), "rmem"
            ]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(32, 8, 8, 4), "f32", ((32, 8, 8, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(32, 8, 8, 4), "f32", (32, 8, 8, 4), "rmem"]
        )
        gw = T.alloc_tensor(tensor_type=Tensor[(1, 8, 8, 4), "f32", (1, 8, 8, 4), "rmem"])
        value_7 = T.alloc_tensor(tensor_type=Tensor[(1, 8, 8, 4), "f32", (1, 8, 8, 4), "rmem"])
        value_8 = T.alloc_tensor(tensor_type=Tensor[(1, 8, 8, 4), "bf16", (1, 8, 8, 4), "rmem"])
        rlo = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"])
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        lo = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        cl = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        hi = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        sl = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        rhi = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"])
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        ch = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"]
        )
        sh = T.alloc_tensor(tensor_type=Tensor[(32, 1, 8, 4), "f32", (32, 1, 8, 4), "rmem"])
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(32, 1, 8, 4), "bf16", (32, 1, 8, 4), "rmem"]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:2], layout=(32, 8), names=("d0", "d1")) as scope_1:
                tile = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 16) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 16) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 32, 0:0 + 256]),
                    layout=(32, 256),
                    shape=(32, 256),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[0:0 + 32, 0:0 + 256]),
                    layout=(32, 8, 8, 4),
                    shape=(32, 8, 8, 4),
                )
                with scope_1 as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                        layout=(32 @ threads.d0, 8, 8 @ threads.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.copy(tile_1, dst_frame)
                for j in range(1, 4, 1):
                    tile_2 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 16) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32):(((((cta.d0 * 16) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32) + 32, 0:0 + 256]),
                        layout=(32, 256),
                        shape=(32, 256),
                    )
                    tile_3 = T.tensor_view(
                        T.ptr_of(tile_2[0:0 + 32, 0:0 + 256]),
                        layout=(32, 8, 8, 4),
                        shape=(32, 8, 8, 4),
                    )
                    with scope_1 as threads_1:
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(value[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.copy(tile_3, dst_frame_1)
                        lhs_frame = T.tensor_view(
                            T.ptr_of(acc[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(acc_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.binary(lhs_frame, rhs_frame, dst_frame_2, kind=BinaryKind.ADD)
                        acc_1_view = T.tensor_view(
                            T.ptr_of(acc_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.copy(acc_1_view, acc)
                with scope_1 as threads_4:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                        layout=(32 @ threads_4.d0, 8, 8 @ threads_4.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(cs[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                        layout=(32 @ threads_4.d0, 8, 8 @ threads_4.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.cast(src_frame, dst_frame_3, dtype="bf16")
                ns = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(32, 256), shape=(32, 256)
                )
                for _h in range(cta.d0 - (2 * (cta.d0 // 2)), 0, 2):
                    with scope_1 as threads_5:
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(cs[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(x32[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.cast(src_frame_1, dst_frame_4, dtype="f32")
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(x32[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(x32[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.binary(lhs_frame_1, rhs_frame_1, dst_frame_5, kind=BinaryKind.MUL)
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_6 = T.tensor_view(
                            T.ptr_of(sq[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=((32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4), (0, 4, 0, 1)),
                            shape=(32, 8, 8, 4),
                        )
                        T.copy(src_frame_2, dst_frame_6)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(sq[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=((32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4), (0, 4, 0, 1)),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_7 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=((32 @ threads_5.d0, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)),
                            shape=(32, 1, 1, 1),
                        )
                        T.reduce(
                            src_frame_3,
                            dst_frame_7,
                            axes=(1, 2, 3),
                            keepdim=True,
                            kind=ReduceKind.MEAN,
                        )
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=((32 @ threads_5.d0, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)),
                            shape=(32, 1, 1, 1),
                        )
                        dst_frame_8 = T.tensor_view(
                            T.ptr_of(ms[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_5.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        T.copy(src_frame_4, dst_frame_8)
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 1e-06)
                    with scope_1 as threads_10:
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(ms[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_10.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        dst_frame_9 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_10.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        T.binary(lhs_frame_2, scalar, dst_frame_9, kind=BinaryKind.ADD)
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_10.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_10.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        T.unary(src_frame_5, dst_frame_10, kind=UnaryKind.RSQRT)
                        lhs_frame_3 = T.tensor_view(
                            T.ptr_of(x32[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_10.d0, 8, 8 @ threads_10.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 32, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                            layout=(32 @ threads_10.d0, 1, 1, 1),
                            shape=(32, 1, 1, 1),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_10.d0, 8, 8 @ threads_10.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.binary(lhs_frame_3, rhs_frame_2, dst_frame_11, kind=BinaryKind.MUL)
                    tile_4 = T.tensor_view(
                        T.ptr_of(gamma_k[0:0 + 256]), layout=(1, 8, 8, 4), shape=(1, 8, 8, 4)
                    )
                    with scope_1 as threads_13:
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_13.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        T.copy(tile_4, dst_frame_12)
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_13.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_13.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        T.cast(src_frame_6, dst_frame_13, dtype="f32")
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 1.0)
                    with scope_1 as threads_15:
                        lhs_frame_4 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_15.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(gw[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_15.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        T.binary(lhs_frame_4, scalar_1, dst_frame_14, kind=BinaryKind.ADD)
                        lhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_15.d0, 8, 8 @ threads_15.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(gw[0:0 + 1, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(1, 8, 8 @ threads_15.d1, 4),
                            shape=(1, 8, 8, 4),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_15.d0, 8, 8 @ threads_15.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.binary(lhs_frame_5, rhs_frame_3, dst_frame_15, kind=BinaryKind.MUL)
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_15.d0, 8, 8 @ threads_15.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(nrm[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_15.d0, 8, 8 @ threads_15.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.cast(src_frame_7, dst_frame_16, dtype="bf16")
                    tile_5 = T.tensor_view(
                        T.ptr_of(nrm[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                        layout=(32, 8, 8, 4),
                        shape=(32, 256),
                    )
                    with scope_1 as threads_18:
                        T.copy(tile_5, ns)
                    tile_6 = T.tensor_view(
                        T.ptr_of(ns[0:0 + 32, 0:0 + 32]),
                        layout=((32, 32), (256, 1)),
                        shape=(32, 32),
                    )
                    tile_7 = T.tensor_view(
                        T.ptr_of(tile_6[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4),
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_19:
                        dst_frame_17 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_19.d0, 1, 8 @ threads_19.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_7, dst_frame_17)
                        src_frame_8 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_19.d0, 1, 8 @ threads_19.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_18 = T.tensor_view(
                            T.ptr_of(lo[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_19.d0, 1, 8 @ threads_19.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_8, dst_frame_18, dtype="f32")
                    tile_8 = T.tensor_view(
                        T.ptr_of(cos[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, 0:0 + 32]),
                        layout=((32, 32), (64, 1)),
                        shape=(32, 32),
                    )
                    tile_9 = T.tensor_view(
                        T.ptr_of(tile_8[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4),
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_21:
                        dst_frame_19 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_9, dst_frame_19)
                        src_frame_9 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_20 = T.tensor_view(
                            T.ptr_of(cl[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_9, dst_frame_20, dtype="f32")
                        lhs_frame_6 = T.tensor_view(
                            T.ptr_of(lo[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_4 = T.tensor_view(
                            T.ptr_of(cl[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_21 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_6, rhs_frame_4, dst_frame_21, kind=BinaryKind.MUL)
                        src_frame_10 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_22 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_10, dst_frame_22, dtype="bf16")
                        src_frame_11 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_23 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_21.d0, 1, 8 @ threads_21.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_11, dst_frame_23, dtype="f32")
                    tile_10 = T.tensor_view(
                        T.ptr_of(ns[0:0 + 32, 32:32 + 32]),
                        layout=((32, 32), (256, 1)) + 32,
                        shape=(32, 32),
                    )
                    tile_11 = T.tensor_view(
                        T.ptr_of(tile_10[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4) + 32,
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_26:
                        dst_frame_24 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_26.d0, 1, 8 @ threads_26.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_11, dst_frame_24)
                        src_frame_12 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_26.d0, 1, 8 @ threads_26.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_25 = T.tensor_view(
                            T.ptr_of(hi[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_26.d0, 1, 8 @ threads_26.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_12, dst_frame_25, dtype="f32")
                        src_frame_13 = T.tensor_view(
                            T.ptr_of(hi[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_26.d0, 1, 8 @ threads_26.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_26 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_26.d0, 1, 8 @ threads_26.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.unary(src_frame_13, dst_frame_26, kind=UnaryKind.NEG)
                    tile_12 = T.tensor_view(
                        T.ptr_of(sin[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, 0:0 + 32]),
                        layout=((32, 32), (64, 1)),
                        shape=(32, 32),
                    )
                    tile_13 = T.tensor_view(
                        T.ptr_of(tile_12[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4),
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_29:
                        dst_frame_27 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_13, dst_frame_27)
                        src_frame_14 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_28 = T.tensor_view(
                            T.ptr_of(sl[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_14, dst_frame_28, dtype="f32")
                        lhs_frame_7 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_5 = T.tensor_view(
                            T.ptr_of(sl[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_29 = T.tensor_view(
                            T.ptr_of(value_17[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_7, rhs_frame_5, dst_frame_29, kind=BinaryKind.MUL)
                        src_frame_15 = T.tensor_view(
                            T.ptr_of(value_17[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_30 = T.tensor_view(
                            T.ptr_of(value_16[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_15, dst_frame_30, dtype="bf16")
                        src_frame_16 = T.tensor_view(
                            T.ptr_of(value_16[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_31 = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_16, dst_frame_31, dtype="f32")
                        lhs_frame_8 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_6 = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_32 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_8, rhs_frame_6, dst_frame_32, kind=BinaryKind.ADD)
                        src_frame_17 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_33 = T.tensor_view(
                            T.ptr_of(rlo[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_29.d0, 1, 8 @ threads_29.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_17, dst_frame_33, dtype="bf16")
                    tile_14 = T.tensor_view(
                        T.ptr_of(rlo[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                        layout=(32, 8, 4),
                        shape=(32, 32),
                    )
                    window = T.tensor_view(
                        T.ptr_of(ns[0:0 + 32, 0:0 + 32]),
                        layout=((32, 32), (256, 1)),
                        shape=(32, 32),
                    )
                    T.copy(tile_14, window)
                    tile_15 = T.tensor_view(
                        T.ptr_of(cos[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, 32:32 + 32]),
                        layout=((32, 32), (64, 1)),
                        shape=(32, 32),
                    )
                    tile_16 = T.tensor_view(
                        T.ptr_of(tile_15[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4),
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_36:
                        dst_frame_34 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_16, dst_frame_34)
                        src_frame_18 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_35 = T.tensor_view(
                            T.ptr_of(ch[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_18, dst_frame_35, dtype="f32")
                        lhs_frame_9 = T.tensor_view(
                            T.ptr_of(hi[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_7 = T.tensor_view(
                            T.ptr_of(ch[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_36 = T.tensor_view(
                            T.ptr_of(value_24[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_9, rhs_frame_7, dst_frame_36, kind=BinaryKind.MUL)
                        src_frame_19 = T.tensor_view(
                            T.ptr_of(value_24[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_37 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_19, dst_frame_37, dtype="bf16")
                        src_frame_20 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_38 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_36.d0, 1, 8 @ threads_36.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_20, dst_frame_38, dtype="f32")
                    tile_17 = T.tensor_view(
                        T.ptr_of(sin[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, 32:32 + 32]),
                        layout=((32, 32), (64, 1)),
                        shape=(32, 32),
                    )
                    tile_18 = T.tensor_view(
                        T.ptr_of(tile_17[0:0 + 32, 0:0 + 32]),
                        layout=(32, 1, 8, 4),
                        shape=(32, 1, 8, 4),
                    )
                    with scope_1 as threads_41:
                        dst_frame_39 = T.tensor_view(
                            T.ptr_of(value_29[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.copy(tile_18, dst_frame_39)
                        src_frame_21 = T.tensor_view(
                            T.ptr_of(value_29[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_40 = T.tensor_view(
                            T.ptr_of(sh[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_21, dst_frame_40, dtype="f32")
                        lhs_frame_10 = T.tensor_view(
                            T.ptr_of(lo[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_8 = T.tensor_view(
                            T.ptr_of(sh[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_41 = T.tensor_view(
                            T.ptr_of(value_28[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_10, rhs_frame_8, dst_frame_41, kind=BinaryKind.MUL)
                        src_frame_22 = T.tensor_view(
                            T.ptr_of(value_28[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_42 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_22, dst_frame_42, dtype="bf16")
                        src_frame_23 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_43 = T.tensor_view(
                            T.ptr_of(value_26[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_23, dst_frame_43, dtype="f32")
                        lhs_frame_11 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        rhs_frame_9 = T.tensor_view(
                            T.ptr_of(value_26[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_44 = T.tensor_view(
                            T.ptr_of(value_21[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.binary(lhs_frame_11, rhs_frame_9, dst_frame_44, kind=BinaryKind.ADD)
                        src_frame_24 = T.tensor_view(
                            T.ptr_of(value_21[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        dst_frame_45 = T.tensor_view(
                            T.ptr_of(rhi[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_41.d0, 1, 8 @ threads_41.d1, 4),
                            shape=(32, 1, 8, 4),
                        )
                        T.cast(src_frame_24, dst_frame_45, dtype="bf16")
                    tile_19 = T.tensor_view(
                        T.ptr_of(rhi[0:0 + 32, 0:0 + 1, 0:0 + 8, 0:0 + 4]),
                        layout=(32, 8, 4),
                        shape=(32, 32),
                    )
                    window_1 = T.tensor_view(
                        T.ptr_of(ns[0:0 + 32, 32:32 + 32]),
                        layout=((32, 32), (256, 1)) + 32,
                        shape=(32, 32),
                    )
                    T.copy(tile_19, window_1)
                    tile_20 = T.tensor_view(
                        T.ptr_of(ns[0:0 + 32, 0:0 + 256]),
                        layout=(32, 8, 8, 4),
                        shape=(32, 8, 8, 4),
                    )
                    with scope_1 as threads_48:
                        dst_frame_46 = T.tensor_view(
                            T.ptr_of(cs_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_48.d0, 8, 8 @ threads_48.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.copy(tile_20, dst_frame_46)
                        cs_1_view = T.tensor_view(
                            T.ptr_of(cs_1[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                            layout=(32 @ threads_48.d0, 8, 8 @ threads_48.d1, 4),
                            shape=(32, 8, 8, 4),
                        )
                        T.copy(cs_1_view, cs)
                tile_21 = T.tensor_view(
                    T.ptr_of(cs[0:0 + 32, 0:0 + 8, 0:0 + 8, 0:0 + 4]),
                    layout=(32, 8, 8, 4),
                    shape=(32, 256),
                )
                window_2 = T.tensor_view(
                    T.ptr_of(out[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, (54 + cta.d0) * 256:(54 + cta.d0) * 256 + 256]),
                    layout=((32, 256), (14336, 1)),
                    shape=(32, 256),
                )
                T.copy(tile_21, window_2)
        T.copy(out, out_1)
