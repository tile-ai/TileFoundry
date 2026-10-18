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
    with Mesh((Topology("cta", 112),), layout=(7, 8, 2), names=("tb", "mi", "cy")) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            with Mesh(
                (Topology("thread", 384),), layout=(64, 4), names=("row", "col")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 16, 4, 4),
                        "f32",
                        (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        "rmem",
                    ]
                )
                value_1 = T.tensor_view(T.ptr_of(ws), layout=(14336, 256), shape=(14336, 256))
                value_2 = T.tensor_view(
                    T.ptr_of(value_1[((((cta.tb * 8) + cta.mi) * 2) * 128) + (cta.cy * 64):((((cta.tb * 8) + cta.mi) * 2) * 128) + (cta.cy * 64) + 1, 0:0 + 1]),
                    layout=(64, 256),
                    shape=(64, 256),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_2), layout=(64, 16, 4, 4), shape=(64, 16, 4, 4)
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                T.copy(value_3, dst_view)
                for j in range(1, 2, 1):
                    value_5 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_6 = T.tensor_view(
                        T.ptr_of(ws), layout=(14336, 256), shape=(14336, 256)
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6[(((((cta.tb * 8) + cta.mi) * 2) + j) * 128) + (cta.cy * 64):(((((cta.tb * 8) + cta.mi) * 2) + j) * 128) + (cta.cy * 64) + 1, 0:0 + 1]),
                        layout=(64, 256),
                        shape=(64, 256),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7), layout=(64, 16, 4, 4), shape=(64, 16, 4, 4)
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_5),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    dst_view_1 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_8, dst_view_1)
                    value_10 = T.tensor_view(
                        T.ptr_of(value),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_10),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.binary(lhs_view, rhs_view, value_12, kind=BinaryKind.ADD)
                cs = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 16, 4, 4),
                        "bf16",
                        (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        "rmem",
                    ]
                )
                value_13 = T.tensor_view(
                    T.ptr_of(value_4),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                value_14 = T.tensor_view(
                    T.ptr_of(value_13),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_14[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                T.cast(src_view, cs, dtype="bf16")
                for _h in range(cta.tb - (7 * (cta.tb // 7)), 3, 7):
                    x32 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(cs),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.cast(value_16, x32, dtype="f32")
                    value_17 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.row, 1, 1, 1), "rmem"
                        ]
                    )
                    value_18 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            ((64 @ scope_1.row, 16, 4 @ scope_1.col, 4), (0, 4, 0, 1)),
                            "rmem",
                        ]
                    )
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, x32, value_19, kind=BinaryKind.MUL)
                    value_20 = T.tensor_view(
                        T.ptr_of(value_18),
                        layout=((64 @ scope_1.row, 16, 4 @ scope_1.col, 4), (0, 4, 0, 1)),
                        shape=(64, 16, 4, 4),
                    )
                    dst_view_2 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=((64 @ scope_1.row, 16, 4 @ scope_1.col, 4), (0, 4, 0, 1)),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_19, dst_view_2)
                    value_21 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1),
                            "f32",
                            ((64 @ scope_1.row, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    src_view_1 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=((64 @ scope_1.row, 16, 4 @ scope_1.col, 4), (0, 4, 0, 1)),
                        shape=(64, 16, 4, 4),
                    )
                    T.reduce(
                        src_view_1,
                        value_21,
                        axes=(1, 2, 3),
                        keepdim=True,
                        kind=ReduceKind.MEAN,
                    )
                    value_22 = T.tensor_view(
                        T.ptr_of(value_17),
                        layout=(64 @ scope_1.row, 1, 1, 1),
                        shape=(64, 1, 1, 1),
                    )
                    dst_view_3 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 64, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(64 @ scope_1.row, 1, 1, 1),
                        shape=(64, 1, 1, 1),
                    )
                    T.copy(value_21, dst_view_3)
                    value_23 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.row, 1, 1, 1), "rmem"
                        ]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 1e-06)
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 64, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(64 @ scope_1.row, 1, 1, 1),
                        shape=(64, 1, 1, 1),
                    )
                    T.binary(lhs_view_1, scalar, value_23, kind=BinaryKind.ADD)
                    value_24 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1, 1, 1), "f32", (64 @ scope_1.row, 1, 1, 1), "rmem"
                        ]
                    )
                    T.unary(value_23, value_24, kind=UnaryKind.RSQRT)
                    value_25 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(x32, value_24, value_25, kind=BinaryKind.MUL)
                    value_26 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "bf16", (1, 16, 4 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    value_27 = T.tensor_view(T.ptr_of(gamma_k), layout=(256,), shape=(256,))
                    value_28 = T.tensor_view(
                        T.ptr_of(value_27),
                        layout=((1, 16, 4, 4), (0, 16, 4, 1)),
                        shape=(1, 16, 4, 4),
                    )
                    value_29 = T.tensor_view(
                        T.ptr_of(value_26),
                        layout=(1, 16, 4 @ scope_1.col, 4),
                        shape=(1, 16, 4, 4),
                    )
                    dst_view_4 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 1, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(1, 16, 4 @ scope_1.col, 4),
                        shape=(1, 16, 4, 4),
                    )
                    T.copy(value_28, dst_view_4)
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "f32", (1, 16, 4 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    src_view_2 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 1, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(1, 16, 4 @ scope_1.col, 4),
                        shape=(1, 16, 4, 4),
                    )
                    T.cast(src_view_2, value_30, dtype="f32")
                    gw = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16, 4, 4), "f32", (1, 16, 4 @ scope_1.col, 4), "rmem"
                        ]
                    )
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 1.0)
                    T.binary(value_30, scalar_1, gw, kind=BinaryKind.ADD)
                    value_31 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_25, gw, value_31, kind=BinaryKind.MUL)
                    nrm = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 16, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_31, nrm, dtype="bf16")
                    value_32 = T.tensor_view(
                        T.ptr_of(nrm),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 256),
                    )
                    value_15 = T.tensor_view(
                        0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                    )
                    value_33 = T.tensor_view(
                        T.ptr_of(value_15), layout=(64, 256), shape=(64, 256)
                    )
                    T.copy(value_32, value_33)
                    value_34 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_35 = T.tensor_view(
                        T.ptr_of(value_33), layout=((64, 32), (256, 1)), shape=(64, 32)
                    )
                    value_36 = T.tensor_view(
                        T.ptr_of(value_35),
                        layout=((64, 2, 4, 4), (256, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_37 = T.tensor_view(
                        T.ptr_of(value_34),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_5 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_36, dst_view_5)
                    lo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_3 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_3, lo, dtype="f32")
                    value_38 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_39 = T.tensor_view(T.ptr_of(cos), layout=(1024, 64), shape=(1024, 64))
                    value_40 = T.tensor_view(
                        T.ptr_of(value_39[(((0 * 8) + cta.mi) * 128) + (cta.cy * 64):(((0 * 8) + cta.mi) * 128) + (cta.cy * 64) + 1, 0:0 + 1]),
                        layout=((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    value_41 = T.tensor_view(
                        T.ptr_of(value_40),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_42 = T.tensor_view(
                        T.ptr_of(value_38),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_6 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_41, dst_view_6)
                    cl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_4 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_4, cl, dtype="f32")
                    value_43 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, cl, value_43, kind=BinaryKind.MUL)
                    value_44 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_43, value_44, dtype="bf16")
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_44, value_45, dtype="f32")
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_47 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 1, 32:32 + 1]),
                        layout=((64, 32), (256, 1)),
                        shape=(64, 32),
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(value_47),
                        layout=((64, 2, 4, 4), (256, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_49 = T.tensor_view(
                        T.ptr_of(value_46),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_7 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_48, dst_view_7)
                    hi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_5 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_5, hi, dtype="f32")
                    value_50 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.unary(hi, value_50, kind=UnaryKind.NEG)
                    value_51 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_52 = T.tensor_view(T.ptr_of(sin), layout=(1024, 64), shape=(1024, 64))
                    value_53 = T.tensor_view(
                        T.ptr_of(value_52[(((0 * 8) + cta.mi) * 128) + (cta.cy * 64):(((0 * 8) + cta.mi) * 128) + (cta.cy * 64) + 1, 0:0 + 1]),
                        layout=((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    value_54 = T.tensor_view(
                        T.ptr_of(value_53),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_55 = T.tensor_view(
                        T.ptr_of(value_51),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_8 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_54, dst_view_8)
                    sl = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_6 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_6, sl, dtype="f32")
                    value_56 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_50, sl, value_56, kind=BinaryKind.MUL)
                    value_57 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_56, value_57, dtype="bf16")
                    value_58 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_57, value_58, dtype="f32")
                    value_59 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_45, value_58, value_59, kind=BinaryKind.ADD)
                    rlo = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_59, rlo, dtype="bf16")
                    value_60 = T.tensor_view(
                        T.ptr_of(rlo),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 32),
                    )
                    value_61 = T.tensor_view(
                        T.ptr_of(value_33), layout=(64, 256), shape=(64, 256)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 64, 0:0 + 32]),
                        layout=((64, 32), (256, 1)),
                        shape=(64, 32),
                    )
                    T.copy(value_60, window)
                    value_62 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_63 = T.tensor_view(
                        T.ptr_of(value_39[(((0 * 8) + cta.mi) * 128) + (cta.cy * 64):(((0 * 8) + cta.mi) * 128) + (cta.cy * 64) + 1, 32:32 + 1]),
                        layout=((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    value_64 = T.tensor_view(
                        T.ptr_of(value_63),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(value_62),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_9 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_64, dst_view_9)
                    ch = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_7 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_7, ch, dtype="f32")
                    value_66 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(hi, ch, value_66, kind=BinaryKind.MUL)
                    value_67 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_66, value_67, dtype="bf16")
                    value_68 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_67, value_68, dtype="f32")
                    value_69 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    value_70 = T.tensor_view(
                        T.ptr_of(value_52[(((0 * 8) + cta.mi) * 128) + (cta.cy * 64):(((0 * 8) + cta.mi) * 128) + (cta.cy * 64) + 1, 32:32 + 1]),
                        layout=((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    value_71 = T.tensor_view(
                        T.ptr_of(value_70),
                        layout=((64, 2, 4, 4), (64, 16, 4, 1)),
                        shape=(64, 2, 4, 4),
                    )
                    value_72 = T.tensor_view(
                        T.ptr_of(value_69),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    dst_view_10 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.copy(value_71, dst_view_10)
                    sh = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    src_view_8 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 64, 0:0 + 2, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 2, 4, 4),
                    )
                    T.cast(src_view_8, sh, dtype="f32")
                    value_73 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(lo, sh, value_73, kind=BinaryKind.MUL)
                    value_74 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_73, value_74, dtype="bf16")
                    value_75 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_74, value_75, dtype="f32")
                    value_76 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "f32",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.binary(value_68, value_75, value_76, kind=BinaryKind.ADD)
                    rhi = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 2, 4, 4),
                            "bf16",
                            (64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                            "rmem",
                        ]
                    )
                    T.cast(value_76, rhi, dtype="bf16")
                    value_77 = T.tensor_view(
                        T.ptr_of(rhi),
                        layout=(64 @ scope_1.row, 2, 4 @ scope_1.col, 4),
                        shape=(64, 32),
                    )
                    value_78 = T.tensor_view(
                        T.ptr_of(value_61), layout=(64, 256), shape=(64, 256)
                    )
                    window_1 = T.tensor_view(
                        T.ptr_of(value_78[0:0 + 64, 32:32 + 32]),
                        layout=((64, 32), (256, 1)) + 32,
                        shape=(64, 32),
                    )
                    T.copy(value_77, window_1)
                    value_79 = T.tensor_view(
                        T.ptr_of(value_78), layout=(64, 16, 4, 4), shape=(64, 16, 4, 4)
                    )
                    value_80 = T.tensor_view(
                        T.ptr_of(value_16),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    value_81 = T.tensor_view(
                        T.ptr_of(value_80),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    dst_view_11 = T.tensor_view(
                        T.ptr_of(value_81[0:0 + 64, 0:0 + 16, 0:0 + 4, 0:0 + 4]),
                        layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                        shape=(64, 16, 4, 4),
                    )
                    T.copy(value_79, dst_view_11)
                value_82 = T.tensor_view(
                    T.ptr_of(cs),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                value_83 = T.tensor_view(
                    T.ptr_of(value_82),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 16, 4, 4),
                )
                value_84 = T.tensor_view(
                    T.ptr_of(value_83),
                    layout=(64 @ scope_1.row, 16, 4 @ scope_1.col, 4),
                    shape=(64, 256),
                )
                value_85 = T.tensor_view(
                    T.ptr_of(out_1), layout=(1024, 14336), shape=(1024, 14336)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_85[(((0 * 8) + cta.mi) * 128) + (cta.cy * 64):(((0 * 8) + cta.mi) * 128) + (cta.cy * 64) + 64, (49 + cta.tb) * 256:(49 + cta.tb) * 256 + 256]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                T.copy(value_84, window_2)
