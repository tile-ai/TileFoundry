# analysis target=nvidia.h200_sxm module=QKVTail function=tail topology=cta wave=112/112
# selection requested=memory executed=memory
# memory traffic=gmem:r144.50KB/w32.00KB@logical,r15.80MB/w3.50MB@total,r144.50KB/w32.00KB@cta,r704B/w128B@thread;rmem:r939.82KB/w795.56KB@logical,r102.79MB/w87.01MB@total,r939.82KB/w795.56KB@cta,r4.62KB/w3.80KB@thread;smem:r40.00KB/w40.00KB@logical,r4.38MB/w4.38MB@total,r40.00KB/w40.00KB@cta,r160B/w160B@thread footprint=cos:128.00KB;gamma_k:512B;sin:128.00KB;v109:781:3.50MB;ws:14.00MB footprint-precision=exact peak=gmem:42.25MB;rmem:224.02KB;smem:32.00KB persistent=gmem:42.25MB
#   buffer=cos holds=17.75MB time=none space=cta.tb reuse=768.00KB fits=yes precision=exact
#   buffer=sin holds=17.75MB time=none space=cta.tb reuse=768.00KB fits=yes precision=exact
#   buffer=gamma_k holds=17.75MB time=none space=cta.tb,cta.mi,cta.cy reuse=55.50KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(1024, 14336), "bf16"], ws: Tensor[(14336, 256), "f32"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(1024, 64), "bf16"], sin: Tensor[(1024, 64), "bf16"], out_1: Tensor[(1024, 14336), "bf16"]
):
    with Mesh((Topology("cta", 112),), layout=(7, 8, 2), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(64, 4), names=("d0", "d1")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 16, 4, 4), "f32", (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 2) * 128) + (cta.d2 * 64):((((cta.d0 * 8) + cta.d1) * 2) * 128) + (cta.d2 * 64) + 1, 0:0 + 1]),
                    layout=(64, 16, 4, 4),
                    shape=(64, 16, 4, 4),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                    shape=(64, 16, 4, 4),
                )
                T.copy(value_1, value_2)
                for j in range(1, 2, 1):
                    value_3 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 2) + j) * 128) + (cta.d2 * 64):(((((cta.d0 * 8) + cta.d1) * 2) + j) * 128) + (cta.d2 * 64) + 1, 0:0 + 1]),
                        layout=(64, 16, 4, 4),
                        shape=(64, 16, 4, 4),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(value_3),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_4, value_5)
                    value_6 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 16, 4, 4),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.binary(value_6, value_5, value_7, kind=BinaryKind.ADD)
                cs = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 16, 4, 4), "bf16", (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4), "rmem"
                    ]
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                    shape=(64, 16, 4, 4),
                )
                T.cast(value_8, cs, dtype="bf16")
                for _h in range(cta.d0 - (7 * (cta.d0 // 7)), 3, 7):
                    x32 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(cs),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.cast(value_10, x32, dtype="f32")
                    value_11 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    value_12 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            ((64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4), (0, 4, 0, 1)),
                            "rmem",
                        ]
                    )
                    value_13 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, x32, value_13, kind=BinaryKind.MUL)
                    value_14 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=((64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4), (0, 4, 0, 1)),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_13, value_14)
                    value_15 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1),
                            "f32",
                            ((64 @ scope_1.d0, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)),
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
                        layout=(64 @ scope_1.d0, 1, 1, 1),
                        shape=(64, 1, 1, 1),
                    )
                    T.copy(value_15, value_16)
                    value_17 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 1e-06)
                    T.binary(value_16, scalar, value_17, kind=BinaryKind.ADD)
                    value_18 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.d0, 1, 1, 1), "rmem"
                        ]
                    )
                    T.unary(value_17, value_18, kind=UnaryKind.RSQRT)
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, value_18, value_19, kind=BinaryKind.MUL)
                    value_20 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "bf16", (1, 16, 4 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    value_21 = T.tensor_view(
                        T.ptr_of(gamma_k),
                        layout=((1, 16, 4, 4), (0, 16, 4, 1)),
                        shape=(1, 16, 4, 4),
                    )
                    value_22 = T.tensor_view(
                        T.ptr_of(value_20),
                        layout=(1, 16, 4 @ scope_1.d1, 4),
                        shape=(1, 16, 4, 4),
                    )
                    T.copy(value_21, value_22)
                    value_23 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "f32", (1, 16, 4 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    T.cast(value_22, value_23, dtype="f32")
                    gw = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "f32", (1, 16, 4 @ scope_1.d1, 4), "rmem"
                        ]
                    )
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 1.0)
                    T.binary(value_23, scalar_1, gw, kind=BinaryKind.ADD)
                    value_24 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_19, gw, value_24, kind=BinaryKind.MUL)
                    nrm = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_24, nrm, dtype="bf16")
                    value_25 = T.tensor_view(
                        T.ptr_of(nrm),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 256),
                    )
                    value_9 = T.tensor_view(
                        0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                    )
                    value_26 = T.tensor_view(
                        T.ptr_of(value_9), layout=(64, 256), shape=(64, 256)
                    )
                    T.copy(value_25, value_26)
                    value_27 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_28 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=((64, 2, 4, 4), (256, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_29 = T.tensor_view(
                        T.ptr_of(value_27),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_28, value_29)
                    lo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_29, lo, dtype="f32")
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_31 = T.tensor_view(
                        T.ptr_of(cos[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64) + 1, 0:0 + 1]),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_32 = T.tensor_view(
                        T.ptr_of(value_30),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_31, value_32)
                    cl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_32, cl, dtype="f32")
                    value_33 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, cl, value_33, kind=BinaryKind.MUL)
                    value_34 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_33, value_34, dtype="bf16")
                    value_35 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_34, value_35, dtype="f32")
                    value_36 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_37 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 1, 32:32 + 1]),
                        layout=((64, 2, 4, 4), (256, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_38 = T.tensor_view(
                        T.ptr_of(value_36),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_37, value_38)
                    hi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_38, hi, dtype="f32")
                    value_39 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.unary(hi, value_39, kind=UnaryKind.NEG)
                    value_40 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_41 = T.tensor_view(
                        T.ptr_of(sin[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64) + 1, 0:0 + 1]),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_42 = T.tensor_view(
                        T.ptr_of(value_40),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_41, value_42)
                    sl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_42, sl, dtype="f32")
                    value_43 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_39, sl, value_43, kind=BinaryKind.MUL)
                    value_44 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_43, value_44, dtype="bf16")
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_44, value_45, dtype="f32")
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_35, value_45, value_46, kind=BinaryKind.ADD)
                    rlo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_46, rlo, dtype="bf16")
                    value_47 = T.tensor_view(
                        T.ptr_of(rlo),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 32),
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(value_9), layout=(64, 256), shape=(64, 256)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 64, 0:0 + 32]),
                        layout=((64, 32), (256, 1)),
                        shape=(64, 32),
                    )
                    T.copy(value_47, window)
                    value_49 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_50 = T.tensor_view(
                        T.ptr_of(cos[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64) + 1, 32:32 + 1]),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_51 = T.tensor_view(
                        T.ptr_of(value_49),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_50, value_51)
                    ch = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_51, ch, dtype="f32")
                    value_52 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(hi, ch, value_52, kind=BinaryKind.MUL)
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_52, value_53, dtype="bf16")
                    value_54 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_53, value_54, dtype="f32")
                    value_55 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    value_56 = T.tensor_view(
                        T.ptr_of(sin[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64) + 1, 32:32 + 1]),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_57 = T.tensor_view(
                        T.ptr_of(value_55),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_56, value_57)
                    sh = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_57, sh, dtype="f32")
                    value_58 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, sh, value_58, kind=BinaryKind.MUL)
                    value_59 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_58, value_59, dtype="bf16")
                    value_60 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_59, value_60, dtype="f32")
                    value_61 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_54, value_60, value_61, kind=BinaryKind.ADD)
                    rhi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_61, rhi, dtype="bf16")
                    value_62 = T.tensor_view(
                        T.ptr_of(rhi),
                        layout=(64 @ scope_1.d0, 2, 4 @ scope_1.d1, 4),
                        shape=(64, 32),
                    )
                    value_63 = T.tensor_view(
                        T.ptr_of(value_9), layout=(64, 256), shape=(64, 256)
                    )
                    window_1 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 64, 32:32 + 32]),
                        layout=((64, 32), (256, 1)) + 32,
                        shape=(64, 32),
                    )
                    T.copy(value_62, window_1)
                    value_64 = T.tensor_view(
                        T.ptr_of(value_9), layout=(64, 16, 4, 4), shape=(64, 16, 4, 4)
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(cs),
                        layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_64, value_65)
                value_66 = T.tensor_view(
                    T.ptr_of(cs),
                    layout=(64 @ scope_1.d0, 16, 4 @ scope_1.d1, 4),
                    shape=(64, 256),
                )
                value_67 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 14336), shape=(1024, 14336)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_67[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 64) + 64, (49 + cta.d0) * 256:(49 + cta.d0) * 256 + 256]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                T.copy(value_66, window_2)
