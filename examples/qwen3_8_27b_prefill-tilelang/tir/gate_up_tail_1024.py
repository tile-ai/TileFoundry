# analysis target=nvidia.h200_sxm module=GateUpTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r128.00KB/w8.00KB@logical,r16.00MB/w1.00MB@total,r128.00KB/w8.00KB@cta,r512B/w32B@thread;rmem:r352.35KB/w336.09KB@logical,r44.04MB/w42.01MB@total,r352.35KB/w336.09KB@cta,r1.72KB/w1.41KB@thread footprint=v45:366:1.00MB;ws:8.00MB footprint-precision=exact peak=gmem:50.00MB;rmem:48.03KB persistent=gmem:50.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(1024, 17408), "bf16"], ws: Tensor[(16384, 256), "f32"], out_1: Tensor[(1024, 17408), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(4, 8, 4), names=("tb", "mi", "cy")) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(32, 8), names=("row", "col")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(T.ptr_of(ws), layout=(16384, 256), shape=(16384, 256))
                value_2 = T.tensor_view(
                    T.ptr_of(value_1[((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32):((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32) + 1, 0:0 + 1]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 32, 0:0 + 128]),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                T.copy(value_2, dst_view)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_1[((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32):((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32) + 1, 128:128 + 1]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_4),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                dst_view_1 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 32, 0:0 + 128]),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                T.copy(value_5, dst_view_1)
                for j in range(1, 4, 1):
                    value_7 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(ws), layout=(16384, 256), shape=(16384, 256)
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_8[(((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32)) + (j * 128):(((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32)) + (j * 128) + 1, 128:128 + 1]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    dst_view_2 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    T.copy(value_9, dst_view_2)
                    value_11 = T.tensor_view(
                        T.ptr_of(value_4),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    T.binary(lhs_view, rhs_view, value_13, kind=BinaryKind.ADD)
                    value_14 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_8[(((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32)) + (j * 128):(((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32)) + (j * 128) + 1, 0:0 + 1]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    dst_view_3 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    T.copy(value_15, dst_view_3)
                    value_17 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(value_17),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 32, 0:0 + 128]),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    value_19 = T.tensor_view(
                        T.ptr_of(value_18),
                        layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                        shape=(32, 128),
                    )
                    T.binary(lhs_view_1, rhs_view_1, value_19, kind=BinaryKind.ADD)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 32, 0:0 + 128]),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                T.cast(src_view, value_20, dtype="bf16")
                g32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.cast(value_20, g32, dtype="f32")
                value_24 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.unary(g32, value_24, kind=UnaryKind.NEG)
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.unary(value_24, value_25, kind=UnaryKind.EXP)
                value_26 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(scalar, value_25, value_26, kind=BinaryKind.ADD)
                value_27 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.binary(g32, value_26, value_27, kind=BinaryKind.DIV)
                value_28 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.cast(value_27, value_28, dtype="bf16")
                value_29 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_30 = T.tensor_view(
                    T.ptr_of(value_6),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                value_31 = T.tensor_view(
                    T.ptr_of(value_30),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                value_32 = T.tensor_view(
                    T.ptr_of(value_31),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_32[0:0 + 32, 0:0 + 128]),
                    layout=(32 @ scope_1.row, 4, 8 @ scope_1.col, 4),
                    shape=(32, 128),
                )
                T.cast(src_view_1, value_29, dtype="bf16")
                act = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.row, 4, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                T.binary(value_28, value_29, act, kind=BinaryKind.MUL)
                value_33 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 17408), shape=(1024, 17408)
                )
                window = T.tensor_view(
                    T.ptr_of(value_33[(((0 * 8) + cta.mi) * 128) + (cta.cy * 32):(((0 * 8) + cta.mi) * 128) + (cta.cy * 32) + 32, (132 + cta.tb) * 128:(132 + cta.tb) * 128 + 128]),
                    layout=((32, 128), (17408, 1)),
                    shape=(32, 128),
                )
                T.copy(act, window)
