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
    with Mesh((Topology("cta", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(32, 8), names=("d0", "d1")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                    shape=(32, 128),
                )
                T.copy(value_1, value_2)
                value_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_4 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 1, 128:128 + 1]),
                    layout=((32, 128), (256, 1)),
                    shape=(32, 128),
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                    shape=(32, 128),
                )
                T.copy(value_4, value_5)
                for j in range(1, 4, 1):
                    value_6 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128):(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128) + 1, 128:128 + 1]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    T.copy(value_7, value_8)
                    value_9 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    T.binary(value_9, value_8, value_10, kind=BinaryKind.ADD)
                    value_11 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128):(((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32)) + (j * 128) + 1, 0:0 + 1]),
                        layout=((32, 128), (256, 1)),
                        shape=(32, 128),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    T.copy(value_12, value_13)
                    value_14 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                        shape=(32, 128),
                    )
                    T.binary(value_14, value_13, value_15, kind=BinaryKind.ADD)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_17 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                    shape=(32, 128),
                )
                T.cast(value_17, value_16, dtype="bf16")
                g32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.cast(value_16, g32, dtype="f32")
                value_18 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.unary(g32, value_18, kind=UnaryKind.NEG)
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.unary(value_18, value_19, kind=UnaryKind.EXP)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(scalar, value_19, value_20, kind=BinaryKind.ADD)
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "f32", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.binary(g32, value_20, value_21, kind=BinaryKind.DIV)
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.cast(value_21, value_22, dtype="bf16")
                value_23 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_24 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=(32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4),
                    shape=(32, 128),
                )
                T.cast(value_24, value_23, dtype="bf16")
                act = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 128), "bf16", (32 @ scope_1.d0, 4, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                T.binary(value_22, value_23, act, kind=BinaryKind.MUL)
                value_25 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 17408), shape=(1024, 17408)
                )
                window = T.tensor_view(
                    T.ptr_of(value_25[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 32, (132 + cta.d0) * 128:(132 + cta.d0) * 128 + 128]),
                    layout=((32, 128), (17408, 1)),
                    shape=(32, 128),
                )
                T.copy(act, window)
