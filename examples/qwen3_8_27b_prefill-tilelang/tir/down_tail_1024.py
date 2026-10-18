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
    with Mesh((Topology("cta", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(32, 8), names=("d0", "d1")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "bf16", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(hidden[0:0 + 1, (((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 1, (16 + cta.d0) * 256:(16 + cta.d0) * 256 + 1]),
                    layout=((32, 256), (5120, 1)),
                    shape=(32, 256),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 256),
                )
                T.copy(value_1, value_2)
                value_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "f32", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_4 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                    layout=(32, 256),
                    shape=(32, 256),
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 256),
                )
                T.copy(value_4, value_5)
                for j in range(1, 4, 1):
                    value_6 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 256), "f32", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32):(((((cta.d0 * 8) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                        layout=(32, 256),
                        shape=(32, 256),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 256),
                    )
                    T.copy(value_7, value_8)
                    value_9 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 256),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 256),
                    )
                    T.binary(value_9, value_8, value_10, kind=BinaryKind.ADD)
                value_11 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "bf16", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 256),
                )
                T.cast(value_12, value_11, dtype="bf16")
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 256), "bf16", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.binary(value_2, value_11, y, kind=BinaryKind.ADD)
                value_13 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 5120), shape=(1024, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_13[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 32, (16 + cta.d0) * 256:(16 + cta.d0) * 256 + 256]),
                    layout=((32, 256), (5120, 1)),
                    shape=(32, 256),
                )
                T.copy(y, window)
