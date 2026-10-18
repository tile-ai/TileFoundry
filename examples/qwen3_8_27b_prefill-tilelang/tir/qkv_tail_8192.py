# analysis target=nvidia.h200_sxm module=QKVTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r136.50KB/w16.00KB@logical,r17.06MB/w2.00MB@total,r136.50KB/w16.00KB@cta,r608B/w64B@thread;rmem:r599.35KB/w527.06KB@logical,r74.92MB/w65.88MB@total,r599.35KB/w527.06KB@cta,r3.01KB/w2.44KB@thread;smem:r20.00KB/w20.00KB@logical,r2.50MB/w2.50MB@total,r20.00KB/w20.00KB@cta,r80B/w80B@thread footprint=cos:0;gamma_k:0;sin:0;v109:781:2.00MB;ws:8.00MB footprint-precision=lower_bound peak=gmem:242.00MB;rmem:112.02KB;smem:16.00KB persistent=gmem:242.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(8192, 14336), "bf16"], ws: Tensor[(16384, 256), "f32"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(8192, 64), "bf16"], sin: Tensor[(8192, 64), "bf16"], out_1: Tensor[(8192, 14336), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(2, 16, 4), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(32, 8), names=("d0", "d1")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 8, 8, 4), "f32", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 16) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 16) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                    layout=(32, 8, 8, 4),
                    shape=(32, 8, 8, 4),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 8, 8, 4),
                )
                T.copy(value_1, value_2)
                for j in range(1, 4, 1):
                    value_3 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 16) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32):(((((cta.d0 * 16) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                        layout=(32, 8, 8, 4),
                        shape=(32, 8, 8, 4),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.copy(value_4, value_5)
                    value_6 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.binary(value_6, value_5, value_7, kind=BinaryKind.ADD)
                cs = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 8, 8, 4), "bf16", (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 8, 8, 4),
                )
                T.cast(value_8, cs, dtype="bf16")
                for _h in range(cta.d0 - (2 * (cta.d0 // 2)), 0, 2):
                    x32 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(cs),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.cast(value_10, x32, dtype="f32")
                    value_11 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 1, 1), "f32", (32 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    value_12 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            ((32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), (0, 4, 0, 1)),
                            "rmem",
                        ]
                    )
                    value_13 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, x32, value_13, kind=BinaryKind.MUL)
                    value_14 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=((32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4), (0, 4, 0, 1)),
                        shape=(32, 8, 8, 4),
                    )
                    T.copy(value_13, value_14)
                    value_15 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 1, 1),
                            "f32",
                            ((32 @ scope_1.d0, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(
                        value_14,
                        value_15,
                        axes=(1, 2, 3),
                        keepdim=True,
                        kind=ReduceKind.MEAN,
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=(32 @ scope_1.d0, 1, 1, 1),
                        shape=(32, 1, 1, 1),
                    )
                    T.copy(value_15, value_16)
                    value_17 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 1, 1), "f32", (32 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 1e-06)
                    T.binary(value_16, scalar, value_17, kind=BinaryKind.ADD)
                    value_18 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 1, 1), "f32", (32 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    T.unary(value_17, value_18, kind=UnaryKind.RSQRT)
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, value_18, value_19, kind=BinaryKind.MUL)
                    value_20 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 8, 8, 4), "bf16", (1, 8, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    value_21 = T.tensor_view(
                        T.ptr_of(gamma_k),
                        layout=((1, 8, 8, 4), (0, 32, 4, 1)),
                        shape=(1, 8, 8, 4),
                    )
                    value_22 = T.tensor_view(
                        T.ptr_of(value_20), layout=(1, 8, 8 @ scope_1.d1, 4), shape=(1, 8, 8, 4)
                    )
                    T.copy(value_21, value_22)
                    value_23 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 8, 8, 4), "f32", (1, 8, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    T.cast(value_22, value_23, dtype="f32")
                    gw = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 8, 8, 4), "f32", (1, 8, 8 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 1.0)
                    T.binary(value_23, scalar_1, gw, kind=BinaryKind.ADD)
                    value_24 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_19, gw, value_24, kind=BinaryKind.MUL)
                    nrm = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 8, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_24, nrm, dtype="bf16")
                    value_25 = T.tensor_view(
                        T.ptr_of(nrm),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 256),
                    )
                    value_9 = T.tensor_view(
                        0, dtype="bf16", storage="smem", layout=(32, 256), shape=(32, 256)
                    )
                    value_26 = T.tensor_view(
                        T.ptr_of(value_9), layout=(32, 256), shape=(32, 256)
                    )
                    T.copy(value_25, value_26)
                    value_27 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_28 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=((32, 1, 8, 4), (256, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_29 = T.tensor_view(
                        T.ptr_of(value_27),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_28, value_29)
                    lo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_29, lo, dtype="f32")
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_31 = T.tensor_view(
                        T.ptr_of(cos[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                        layout=((32, 1, 8, 4), (64, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_32 = T.tensor_view(
                        T.ptr_of(value_30),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_31, value_32)
                    cl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_32, cl, dtype="f32")
                    value_33 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, cl, value_33, kind=BinaryKind.MUL)
                    value_34 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_33, value_34, dtype="bf16")
                    value_35 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_34, value_35, dtype="f32")
                    value_36 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_37 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 1, 32:32 + 1]),
                        layout=((32, 1, 8, 4), (256, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_38 = T.tensor_view(
                        T.ptr_of(value_36),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_37, value_38)
                    hi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_38, hi, dtype="f32")
                    value_39 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.unary(hi, value_39, kind=UnaryKind.NEG)
                    value_40 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_41 = T.tensor_view(
                        T.ptr_of(sin[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 1, 0:0 + 1]),
                        layout=((32, 1, 8, 4), (64, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_42 = T.tensor_view(
                        T.ptr_of(value_40),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_41, value_42)
                    sl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_42, sl, dtype="f32")
                    value_43 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_39, sl, value_43, kind=BinaryKind.MUL)
                    value_44 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_43, value_44, dtype="bf16")
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_44, value_45, dtype="f32")
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_35, value_45, value_46, kind=BinaryKind.ADD)
                    rlo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_46, rlo, dtype="bf16")
                    value_47 = T.tensor_view(
                        T.ptr_of(rlo),
                        layout=(32 @ scope_1.d0, 8 @ scope_1.d1, 4),
                        shape=(32, 32),
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(value_9), layout=(32, 256), shape=(32, 256)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 32, 0:0 + 32]),
                        layout=((32, 32), (256, 1)),
                        shape=(32, 32),
                    )
                    T.copy(value_47, window)
                    value_49 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_50 = T.tensor_view(
                        T.ptr_of(cos[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 1, 32:32 + 1]),
                        layout=((32, 1, 8, 4), (64, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_51 = T.tensor_view(
                        T.ptr_of(value_49),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_50, value_51)
                    ch = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_51, ch, dtype="f32")
                    value_52 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(hi, ch, value_52, kind=BinaryKind.MUL)
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_52, value_53, dtype="bf16")
                    value_54 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_53, value_54, dtype="f32")
                    value_55 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_56 = T.tensor_view(
                        T.ptr_of(sin[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 1, 32:32 + 1]),
                        layout=((32, 1, 8, 4), (64, 0, 4, 1)),
                        shape=(32, 1, 8, 4),
                    )
                    value_57 = T.tensor_view(
                        T.ptr_of(value_55),
                        layout=(32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                        shape=(32, 1, 8, 4),
                    )
                    T.copy(value_56, value_57)
                    sh = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_57, sh, dtype="f32")
                    value_58 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, sh, value_58, kind=BinaryKind.MUL)
                    value_59 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_58, value_59, dtype="bf16")
                    value_60 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_59, value_60, dtype="f32")
                    value_61 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "f32",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_54, value_60, value_61, kind=BinaryKind.ADD)
                    rhi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (32, 1, 8, 4),
                            "bf16",
                            (32 @ scope_1.d0, 1, 8 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_61, rhi, dtype="bf16")
                    value_62 = T.tensor_view(
                        T.ptr_of(rhi),
                        layout=(32 @ scope_1.d0, 8 @ scope_1.d1, 4),
                        shape=(32, 32),
                    )
                    value_63 = T.tensor_view(
                        T.ptr_of(value_9), layout=(32, 256), shape=(32, 256)
                    )
                    window_1 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 32, 32:32 + 32]),
                        layout=((32, 32), (256, 1)) + 32,
                        shape=(32, 32),
                    )
                    T.copy(value_62, window_1)
                    value_64 = T.tensor_view(
                        T.ptr_of(value_9), layout=(32, 8, 8, 4), shape=(32, 8, 8, 4)
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(cs),
                        layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                        shape=(32, 8, 8, 4),
                    )
                    T.copy(value_64, value_65)
                value_66 = T.tensor_view(
                    T.ptr_of(cs),
                    layout=(32 @ scope_1.d0, 8, 8 @ scope_1.d1, 4),
                    shape=(32, 256),
                )
                value_67 = T.tensor_view(
                    T.ptr_of(out_1), layout=(8192, 14336), shape=(8192, 14336)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_67[(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32):(((3 * 16) + cta.d1) * 128) + (cta.d2 * 32) + 32, (54 + cta.d0) * 256:(54 + cta.d0) * 256 + 256]),
                    layout=((32, 256), (14336, 1)),
                    shape=(32, 256),
                )
                T.copy(value_66, window_2)
