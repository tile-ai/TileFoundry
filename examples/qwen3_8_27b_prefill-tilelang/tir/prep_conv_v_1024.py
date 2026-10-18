# analysis target=nvidia.h200_sxm module=Prep function=conv_v topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r17.75KB/w12.02MB@logical,r22.19MB/w15.02GB@total,r17.75KB/w12.02MB@cta,r16.11KB/w12.00MB@thread;rmem:r645.01KB/w612.56KB@logical,r806.26MB/w765.71MB@total,r645.01KB/w612.56KB@cta,r5.79KB/w5.10KB@thread;smem:r80.00KB/w49.50KB@logical,r100.00MB/w61.88MB@total,r80.00KB/w49.50KB@cta,r16.50KB/w48.80KB@thread footprint=conv_w:48.00KB;qkv:1.10MB;v92:180:1.06MB footprint-precision=exact peak=gmem:32.08MB;rmem:80.02KB;smem:32.75KB persistent=gmem:20.08MB
#   buffer=conv_w holds=2.21MB time=none space=cta.bt reuse=48.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def conv_v(
    qkv: Tensor[(1, 1024, 10240), "bf16", ((1, 1024, 10240), (17039360, 16640, 1))], conv_w: ConstTensor[(10240, 1, 4), "bf16"], out: Tensor[(128, 8, 48, 16, 8), "bf16"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("bt", "cb")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("s", "g")) as scope:
            value = T.tensor_view(
                T.ptr_of(out), layout=(128, 8, 48, 16, 8), shape=(128, 8, 48, 16, 8)
            )
            T.fill(value, 0.0)
            for h in range((cta.cb - (80 * ((cta.cb - 32) // 80))) - 32, 48, 80):
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                slab = T.tensor_view(
                    16384, dtype="bf16", storage="smem", layout=(67, 16, 8), shape=(67, 16, 8)
                )
                T.fill(slab, 0.0)
                value_2 = T.alloc_tensor(
                    tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16 @ scope.g, 8), "rmem"]
                )
                value_3 = T.tensor_view(
                    T.ptr_of(qkv),
                    layout=((1, 1024, 10240), (17039360, 16640, 1)),
                    shape=(1, 1024, 10240),
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value_3), layout=((1024, 10240), (16640, 1)), shape=(1024, 10240)
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_4), layout=((1024, 10240), (16640, 1)), shape=(1024, 10240)
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_5[(((cta.bt * 64) + 1024) - (4 - 1)) % 1024:(((cta.bt * 64) + 1024) - (4 - 1)) % 1024 + 1, ((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1]),
                    layout=((3, 128), (16640, 1)),
                    shape=(3, 128),
                )
                value_7 = T.tensor_view(
                    T.ptr_of(value_6), layout=((3, 16, 8), (16640, 8, 1)), shape=(3, 16, 8)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value_2), layout=(3, 16 @ scope.g, 8), shape=(3, 16, 8)
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                    layout=(3, 16 @ scope.g, 8),
                    shape=(3, 16, 8),
                )
                T.copy(value_7, dst_view)
                keep = T.alloc_tensor(tensor_type=Tensor[(), "bf16", "rmem"])
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar, ceildiv(cta.bt, 16))
                T.cast(scalar, keep, dtype="bf16")
                value_9 = T.alloc_tensor(
                    tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16 @ scope.g, 8), "rmem"]
                )
                lhs_view = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                    layout=(3, 16 @ scope.g, 8),
                    shape=(3, 16, 8),
                )
                T.binary(lhs_view, keep, value_9, kind=BinaryKind.MUL)
                value_10 = T.tensor_view(T.ptr_of(slab), layout=(67, 16, 8), shape=(67, 16, 8))
                window = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                    layout=(3, 16, 8),
                    shape=(3, 16, 8),
                )
                T.copy(value_9, window)
                value_12 = T.tensor_view(
                    T.ptr_of(value_5[cta.bt * 64:cta.bt * 64 + 1, ((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1]),
                    layout=((64, 128), (16640, 1)),
                    shape=(64, 128),
                )
                value_13 = T.tensor_view(
                    T.ptr_of(value_12), layout=((64, 16, 8), (16640, 8, 1)), shape=(64, 16, 8)
                )
                value_11 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 16, 8), shape=(64, 16, 8)
                )
                value_14 = T.tensor_view(
                    T.ptr_of(value_11), layout=(64, 16, 8), shape=(64, 16, 8)
                )
                T.copy(value_13, value_14)
                value_15 = T.tensor_view(
                    T.ptr_of(value_10), layout=(67, 16, 8), shape=(67, 16, 8)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_15[4 - 1:4 - 1 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                T.copy(value_14, window_1)
                value_16 = T.tensor_view(
                    T.ptr_of(value_15[3:3 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                value_17 = T.tensor_view(
                    T.ptr_of(value_16), layout=(8, 8, 16, 8), shape=(8, 8, 16, 8)
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                dst_view_1 = T.tensor_view(
                    T.ptr_of(value_18[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_17, dst_view_1)
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_18[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.cast(src_view, value_19, dtype="f32")
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                value_21 = T.tensor_view(
                    T.ptr_of(conv_w), layout=(10240, 1, 4), shape=(10240, 1, 4)
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21), layout=(10240, 4), shape=(10240, 4)
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22), layout=(10240, 4), shape=(10240, 4)
                )
                value_24 = T.tensor_view(
                    T.ptr_of(value_23[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1, 3:3 + 1]),
                    layout=((128, 1), (4, 0)),
                    shape=(128, 1),
                )
                value_25 = T.tensor_view(
                    T.ptr_of(value_24),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_26 = T.tensor_view(
                    T.ptr_of(value_20), layout=(1, 1, 16 @ scope.g, 8), shape=(1, 1, 16, 8)
                )
                dst_view_2 = T.tensor_view(
                    T.ptr_of(value_26[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.copy(value_25, dst_view_2)
                value_27 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_26[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.cast(src_view_1, value_27, dtype="f32")
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(value_19, value_27, acc, kind=BinaryKind.MUL)
                value_28 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                value_29 = T.tensor_view(
                    T.ptr_of(value_15[2:2 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                value_30 = T.tensor_view(
                    T.ptr_of(value_29), layout=(8, 8, 16, 8), shape=(8, 8, 16, 8)
                )
                value_31 = T.tensor_view(
                    T.ptr_of(value_28),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                dst_view_3 = T.tensor_view(
                    T.ptr_of(value_31[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_30, dst_view_3)
                value_32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                src_view_2 = T.tensor_view(
                    T.ptr_of(value_31[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.cast(src_view_2, value_32, dtype="f32")
                value_33 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                value_34 = T.tensor_view(
                    T.ptr_of(value_23[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1, 2:2 + 1]),
                    layout=((128, 1), (4, 0)),
                    shape=(128, 1),
                )
                value_35 = T.tensor_view(
                    T.ptr_of(value_34),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_36 = T.tensor_view(
                    T.ptr_of(value_33), layout=(1, 1, 16 @ scope.g, 8), shape=(1, 1, 16, 8)
                )
                dst_view_4 = T.tensor_view(
                    T.ptr_of(value_36[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.copy(value_35, dst_view_4)
                value_37 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                src_view_3 = T.tensor_view(
                    T.ptr_of(value_36[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.cast(src_view_3, value_37, dtype="f32")
                value_38 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(value_32, value_37, value_38, kind=BinaryKind.MUL)
                acc_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(acc, value_38, acc_1, kind=BinaryKind.ADD)
                value_39 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                value_40 = T.tensor_view(
                    T.ptr_of(value_15[1:1 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                value_41 = T.tensor_view(
                    T.ptr_of(value_40), layout=(8, 8, 16, 8), shape=(8, 8, 16, 8)
                )
                value_42 = T.tensor_view(
                    T.ptr_of(value_39),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                dst_view_5 = T.tensor_view(
                    T.ptr_of(value_42[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_41, dst_view_5)
                value_43 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                src_view_4 = T.tensor_view(
                    T.ptr_of(value_42[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.cast(src_view_4, value_43, dtype="f32")
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                value_45 = T.tensor_view(
                    T.ptr_of(value_23[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1, 1:1 + 1]),
                    layout=((128, 1), (4, 0)),
                    shape=(128, 1),
                )
                value_46 = T.tensor_view(
                    T.ptr_of(value_45),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_47 = T.tensor_view(
                    T.ptr_of(value_44), layout=(1, 1, 16 @ scope.g, 8), shape=(1, 1, 16, 8)
                )
                dst_view_6 = T.tensor_view(
                    T.ptr_of(value_47[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.copy(value_46, dst_view_6)
                value_48 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                src_view_5 = T.tensor_view(
                    T.ptr_of(value_47[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.cast(src_view_5, value_48, dtype="f32")
                value_49 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(value_43, value_48, value_49, kind=BinaryKind.MUL)
                acc_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(acc_1, value_49, acc_2, kind=BinaryKind.ADD)
                value_50 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                value_51 = T.tensor_view(
                    T.ptr_of(value_15), layout=(64, 16, 8), shape=(64, 16, 8)
                )
                value_52 = T.tensor_view(
                    T.ptr_of(value_51), layout=(8, 8, 16, 8), shape=(8, 8, 16, 8)
                )
                value_53 = T.tensor_view(
                    T.ptr_of(value_50),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                dst_view_7 = T.tensor_view(
                    T.ptr_of(value_53[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.copy(value_52, dst_view_7)
                value_54 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                src_view_6 = T.tensor_view(
                    T.ptr_of(value_53[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 8),
                    shape=(8, 8, 16, 8),
                )
                T.cast(src_view_6, value_54, dtype="f32")
                value_55 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                value_56 = T.tensor_view(
                    T.ptr_of(value_23[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 1, 0:0 + 1]),
                    layout=((128, 1), (4, 0)),
                    shape=(128, 1),
                )
                value_57 = T.tensor_view(
                    T.ptr_of(value_56),
                    layout=((1, 1, 16, 8), (0, 0, 32, 4)),
                    shape=(1, 1, 16, 8),
                )
                value_58 = T.tensor_view(
                    T.ptr_of(value_55), layout=(1, 1, 16 @ scope.g, 8), shape=(1, 1, 16, 8)
                )
                dst_view_8 = T.tensor_view(
                    T.ptr_of(value_58[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.copy(value_57, dst_view_8)
                value_59 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16 @ scope.g, 8), "rmem"]
                )
                src_view_7 = T.tensor_view(
                    T.ptr_of(value_58[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                    layout=(1, 1, 16 @ scope.g, 8),
                    shape=(1, 1, 16, 8),
                )
                T.cast(src_view_7, value_59, dtype="f32")
                value_60 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(value_54, value_59, value_60, kind=BinaryKind.MUL)
                acc_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(acc_2, value_60, acc_3, kind=BinaryKind.ADD)
                value_61 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.cast(acc_3, value_61, dtype="bf16")
                xb = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.cast(value_61, xb, dtype="f32")
                value_62 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.unary(xb, value_62, kind=UnaryKind.NEG)
                value_63 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.unary(value_62, value_63, kind=UnaryKind.EXP)
                value_64 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(value_63, scalar_1, value_64, kind=BinaryKind.ADD)
                value_65 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "f32", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.binary(xb, value_64, value_65, kind=BinaryKind.DIV)
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 8), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 8), "rmem"
                    ]
                )
                T.cast(value_65, y, dtype="bf16")
                value_66 = T.tensor_view(
                    T.ptr_of(y),
                    layout=(8 @ scope.s, 8, 1, 16 @ scope.g, 8),
                    shape=(8, 8, 1, 16, 8),
                )
                value_67 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 48, 16, 8), shape=(128, 8, 48, 16, 8)
                )
                value_68 = T.tensor_view(
                    T.ptr_of(value_67), layout=(128, 8, 48, 16, 8), shape=(128, 8, 48, 16, 8)
                )
                value_69 = T.tensor_view(
                    T.ptr_of(value_68), layout=(128, 8, 48, 16, 8), shape=(128, 8, 48, 16, 8)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_69[cta.bt * (64 // 8):cta.bt * (64 // 8) + 8, 0:0 + 8, h:h + 1, 0:0 + 16, 0:0 + 8]),
                    layout=((8, 8, 1, 16, 8), (49152, 6144, 128, 8, 1)),
                    shape=(8, 8, 1, 16, 8),
                )
                T.copy(value_66, window_2)
