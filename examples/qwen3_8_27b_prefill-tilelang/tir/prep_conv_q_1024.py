# analysis target=nvidia.h200_sxm module=Prep function=conv_q topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r17.75KB/w4.02MB@logical,r22.19MB/w5.02GB@total,r17.75KB/w4.02MB@cta,r16.11KB/w4.00MB@thread;rmem:r903.12KB/w806.69KB@logical,r1.10GB/w1008.36MB@total,r903.12KB/w806.69KB@cta,r8.04KB/w6.86KB@thread;smem:r80.00KB/w49.50KB@logical,r100.00MB/w61.88MB@total,r80.00KB/w49.50KB@cta,r16.50KB/w48.80KB@thread footprint=conv_w:16.00KB;qkv:524.00KB;v108:101:512.00KB footprint-precision=exact peak=gmem:24.08MB;rmem:96.02KB;smem:32.75KB persistent=gmem:20.08MB
#   buffer=conv_w holds=1.03MB time=none space=cta.bt reuse=16.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def conv_q(
    qkv: Tensor[(1, 1024, 10240), "bf16", ((1, 1024, 10240), (17039360, 16640, 1))], conv_w: ConstTensor[(10240, 1, 4), "bf16"], out: Tensor[(128, 8, 16, 16, 8), "bf16"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            for h in range(cta.d1 - (80 * (cta.d1 // 80)), 16, 80):
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                slab = T.tensor_view(
                    16384, dtype="bf16", storage="smem", layout=(67, 16, 8), shape=(67, 16, 8)
                )
                T.fill(slab, 0.0)
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16 @ scope.d1, 8), "rmem"]
                )
                value_2 = T.tensor_view(
                    T.ptr_of(qkv[0:0 + 1, (((cta.d0 * 64) + 1024) - (4 - 1)) % 1024:(((cta.d0 * 64) + 1024) - (4 - 1)) % 1024 + 1, (0 + h) * 128:(0 + h) * 128 + 1]),
                    layout=((3, 16, 8), (16640, 8, 1)),
                    shape=(3, 16, 8),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_1), layout=(3, 16 @ scope.d1, 8), shape=(3, 16, 8)
                )
                T.copy(value_2, value_3)
                keep = T.alloc_tensor(tensor_type=Tensor[(), "bf16", "rmem"])
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar, ceildiv(cta.d0, 16))
                T.cast(scalar, keep, dtype="bf16")
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16 @ scope.d1, 8), "rmem"]
                )
                T.binary(value_3, keep, value_4, kind=BinaryKind.MUL)
                value_5 = T.tensor_view(T.ptr_of(slab), layout=(67, 16, 8), shape=(67, 16, 8))
                window = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                    layout=(3, 16, 8),
                    shape=(3, 16, 8),
                )
                T.copy(value_4, window)
                value_7 = T.tensor_view(
                    T.ptr_of(qkv[0:0 + 1, cta.d0 * 64:cta.d0 * 64 + 1, (0 + h) * 128:(0 + h) * 128 + 1]),
                    layout=((64, 16, 8), (16640, 8, 1)),
                    shape=(64, 16, 8),
                )
                value_6 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 16, 8), shape=(64, 16, 8)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value_6), layout=(64, 16, 8), shape=(64, 16, 8)
                )
                T.copy(value_7, value_8)
                value_9 = T.tensor_view(T.ptr_of(slab), layout=(67, 16, 8), shape=(67, 16, 8))
                window_1 = T.tensor_view(
                    T.ptr_of(value_9[4 - 1:4 - 1 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                T.copy(value_8, window_1)
                value_10 = T.tensor_view(
                    T.ptr_of(slab[3:3 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8, 8, 16, 8),
                    shape=(8, 8, 16, 8),
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_10, value_11)
                value_12 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_11, value_12, dtype="f32")
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                value_14 = T.tensor_view(
                    T.ptr_of(conv_w[(0 + h) * 128:(0 + h) * 128 + 1, 0:0 + 1, 3:3 + 1]),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_15 = T.tensor_view(
                    T.ptr_of(value_13), layout=(1, 1, 16 @ scope.d1, 8), shape=(1, 1, 16, 8)
                )
                T.copy(value_14, value_15)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                T.cast(value_15, value_16, dtype="f32")
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(value_12, value_16, acc, kind=BinaryKind.MUL)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                value_18 = T.tensor_view(
                    T.ptr_of(slab[2:2 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8, 8, 16, 8),
                    shape=(8, 8, 16, 8),
                )
                value_19 = T.tensor_view(
                    T.ptr_of(value_17),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_18, value_19)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_19, value_20, dtype="f32")
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                value_22 = T.tensor_view(
                    T.ptr_of(conv_w[(0 + h) * 128:(0 + h) * 128 + 1, 0:0 + 1, 2:2 + 1]),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_21), layout=(1, 1, 16 @ scope.d1, 8), shape=(1, 1, 16, 8)
                )
                T.copy(value_22, value_23)
                value_24 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                T.cast(value_23, value_24, dtype="f32")
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(value_20, value_24, value_25, kind=BinaryKind.MUL)
                acc_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(acc, value_25, acc_1, kind=BinaryKind.ADD)
                value_26 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                value_27 = T.tensor_view(
                    T.ptr_of(slab[1:1 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(8, 8, 16, 8),
                    shape=(8, 8, 16, 8),
                )
                value_28 = T.tensor_view(
                    T.ptr_of(value_26),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_27, value_28)
                value_29 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_28, value_29, dtype="f32")
                value_30 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                value_31 = T.tensor_view(
                    T.ptr_of(conv_w[(0 + h) * 128:(0 + h) * 128 + 1, 0:0 + 1, 1:1 + 1]),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_32 = T.tensor_view(
                    T.ptr_of(value_30), layout=(1, 1, 16 @ scope.d1, 8), shape=(1, 1, 16, 8)
                )
                T.copy(value_31, value_32)
                value_33 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                T.cast(value_32, value_33, dtype="f32")
                value_34 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(value_29, value_33, value_34, kind=BinaryKind.MUL)
                acc_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(acc_1, value_34, acc_2, kind=BinaryKind.ADD)
                value_35 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                value_36 = T.tensor_view(
                    T.ptr_of(slab), layout=(8, 8, 16, 8), shape=(8, 8, 16, 8)
                )
                value_37 = T.tensor_view(
                    T.ptr_of(value_35),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_36, value_37)
                value_38 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_37, value_38, dtype="f32")
                value_39 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                value_40 = T.tensor_view(
                    T.ptr_of(conv_w[(0 + h) * 128:(0 + h) * 128 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_41 = T.tensor_view(
                    T.ptr_of(value_39), layout=(1, 1, 16 @ scope.d1, 8), shape=(1, 1, 16, 8)
                )
                T.copy(value_40, value_41)
                value_42 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.d1, 8), "rmem"]
                )
                T.cast(value_41, value_42, dtype="f32")
                value_43 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(value_38, value_42, value_43, kind=BinaryKind.MUL)
                acc_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(acc_2, value_43, acc_3, kind=BinaryKind.ADD)
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(acc_3, value_44, dtype="bf16")
                xb = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_44, xb, dtype="f32")
                value_45 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.unary(xb, value_45, kind=UnaryKind.NEG)
                value_46 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.unary(value_45, value_46, kind=UnaryKind.EXP)
                value_47 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(value_46, scalar_1, value_47, kind=BinaryKind.ADD)
                value_48 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(xb, value_47, value_48, kind=BinaryKind.DIV)
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_48, y, dtype="bf16")
                y32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(y, y32, dtype="f32")
                value_49 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                value_50 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8),
                        "f32",
                        ((8 @ scope.d0, 8, 16 @ scope.d1, 8), (0, 8, 0, 1)),
                        "rmem",
                    ]
                )
                value_51 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(y32, y32, value_51, kind=BinaryKind.MUL)
                value_52 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_51, value_52, dtype="bf16")
                value_53 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_52, value_53, dtype="f32")
                value_54 = T.tensor_view(
                    T.ptr_of(value_50),
                    layout=((8 @ scope.d0, 8, 16 @ scope.d1, 8), (0, 8, 0, 1)),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_53, value_54)
                value_55 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 1, 1),
                        "f32",
                        ((8 @ scope.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_54, value_55, axes=(2, 3), keepdim=True, kind=ReduceKind.SUM)
                value_56 = T.tensor_view(
                    T.ptr_of(value_49), layout=(8 @ scope.d0, 8, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_55, value_56)
                value_57 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "bf16", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_56, value_57, dtype="bf16")
                sc = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_57, sc, dtype="f32")
                value_58 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1e-06)
                T.binary(sc, scalar_2, value_58, kind=BinaryKind.ADD)
                value_59 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "bf16", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_58, value_59, dtype="bf16")
                sc_1 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_59, sc_1, dtype="f32")
                value_60 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.unary(sc_1, value_60, kind=UnaryKind.RSQRT)
                value_61 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "bf16", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_60, value_61, dtype="bf16")
                sc_2 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8 @ scope.d0, 8, 1, 1), "rmem"]
                )
                T.cast(value_61, sc_2, dtype="f32")
                value_62 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.binary(y32, sc_2, value_62, kind=BinaryKind.MUL)
                y_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 8), "rmem"
                    ]
                )
                T.cast(value_62, y_1, dtype="bf16")
                value_63 = T.tensor_view(
                    T.ptr_of(y_1),
                    layout=(8 @ scope.d0, 8, 1, 16 @ scope.d1, 8),
                    shape=(8, 8, 1, 16, 8),
                )
                value_64 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 16, 16, 8), shape=(128, 8, 16, 16, 8)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_64[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, h:h + 1, 0:0 + 16, 0:0 + 8]),
                    layout=((8, 8, 1, 16, 8), (16384, 2048, 128, 8, 1)),
                    shape=(8, 8, 1, 16, 8),
                )
                T.copy(value_63, window_2)
