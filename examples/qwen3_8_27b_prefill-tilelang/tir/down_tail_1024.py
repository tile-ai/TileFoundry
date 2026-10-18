# analysis target=nvidia.h200_sxm module=ResidualTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r144.00KB/w16.00KB@logical,r18.00MB/w2.00MB@total,r144.00KB/w16.00KB@cta,r576B/w64B@thread;rmem:r272.23KB/w272.06KB@logical,r34.03MB/w34.01MB@total,r272.23KB/w272.06KB@cta,r1.30KB/w1.12KB@thread footprint=hidden:2.00MB;v34:212:2.00MB;ws:8.00MB footprint-precision=exact peak=gmem:36.00MB;rmem:80.03KB persistent=gmem:36.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(1024, 5120), "bf16"], ws: Tensor[(16384, 256), "f32"], hidden: Tensor[(1, 1024, 5120), "bf16"], out_1: Tensor[(1024, 5120), "bf16"]
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
                        (32, 256), "bf16", (32 @ scope_1.row, 8, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(hidden), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value_1), layout=(1024, 5120), shape=(1024, 5120)
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_2[(((0 * 8) + cta.mi) * 128) + (cta.cy * 32):(((0 * 8) + cta.mi) * 128) + (cta.cy * 32) + 1, (16 + cta.tb) * 256:(16 + cta.tb) * 256 + 1]),
                    layout=((32, 256), (5120, 1)),
                    shape=(32, 256),
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 32, 0:0 + 256]),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                T.copy(value_3, dst_view)
                value_5 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "f32", (32 @ scope_1.row, 8, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_6 = T.tensor_view(T.ptr_of(ws), layout=(16384, 256), shape=(16384, 256))
                value_7 = T.tensor_view(
                    T.ptr_of(value_6[((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32):((((cta.tb * 8) + cta.mi) * 4) * 128) + (cta.cy * 32) + 1, 0:0 + 1]),
                    layout=(32, 256),
                    shape=(32, 256),
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value_5),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                dst_view_1 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 32, 0:0 + 256]),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                T.copy(value_7, dst_view_1)
                for j in range(1, 4, 1):
                    value_9 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 256), "f32", (32 @ scope_1.row, 8, 8 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(ws), layout=(16384, 256), shape=(16384, 256)
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_10[(((((cta.tb * 8) + cta.mi) * 4) + j) * 128) + (cta.cy * 32):(((((cta.tb * 8) + cta.mi) * 4) + j) * 128) + (cta.cy * 32) + 1, 0:0 + 1]),
                        layout=(32, 256),
                        shape=(32, 256),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    dst_view_2 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    T.copy(value_11, dst_view_2)
                    value_13 = T.tensor_view(
                        T.ptr_of(value_5),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_13),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                        shape=(32, 256),
                    )
                    T.binary(lhs_view, rhs_view, value_15, kind=BinaryKind.ADD)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "bf16", (32 @ scope_1.row, 8, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                value_17 = T.tensor_view(
                    T.ptr_of(value_8),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_17),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_18[0:0 + 32, 0:0 + 256]),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                T.cast(src_view, value_16, dtype="bf16")
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "bf16", (32 @ scope_1.row, 8, 8 @ scope_1.col, 4), "rmem"
                    ]
                )
                lhs_view_1 = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 32, 0:0 + 256]),
                    layout=(32 @ scope_1.row, 8, 8 @ scope_1.col, 4),
                    shape=(32, 256),
                )
                T.binary(lhs_view_1, value_16, y, kind=BinaryKind.ADD)
                value_19 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 5120), shape=(1024, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_19[(((0 * 8) + cta.mi) * 128) + (cta.cy * 32):(((0 * 8) + cta.mi) * 128) + (cta.cy * 32) + 32, (16 + cta.tb) * 256:(16 + cta.tb) * 256 + 256]),
                    layout=((32, 256), (5120, 1)),
                    shape=(32, 256),
                )
                T.copy(y, window)
