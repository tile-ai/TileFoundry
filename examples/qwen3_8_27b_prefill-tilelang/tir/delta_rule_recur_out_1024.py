# analysis target=nvidia.h200_sxm module=DeltaRuleRecur function=recur_out topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r1.13MB/w12.25MB@logical,r108.38MB/w1.15GB@total,r1.13MB/w12.25MB@cta,r1.13MB/w12.25MB@thread;rmem:r4.78MB/w3.59MB@logical,r459.33MB/w389.64MB@total,r4.78MB/w4.06MB@cta,r42.16KB/w33.02KB@thread;smem:r1.89MB/w1.64MB@logical,r181.14MB/w157.12MB@total,r1.89MB/w1.64MB@cta,r349.44KB/w1.01MB@thread footprint=gc:12.00KB;k:256.00KB;q:256.00KB;v95:348:8.00KB;v97:349:768.00KB;wup:1.88MB footprint-precision=exact peak=gmem:50.20MB;rmem:64.27KB;smem:226.00KB persistent=gmem:38.19MB
#   buffer=wup holds=3.14MB time=none space=cta.vb reuse=1.88MB fits=yes precision=exact
#   buffer=k holds=3.14MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=q holds=3.14MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=gc holds=3.14MB time=none space=cta.vb reuse=12.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def recur_out(
    q: Tensor[(1024, 16, 128), "bf16"], k: Tensor[(1024, 16, 128), "bf16"], wup: Tensor[(1024, 48, 320), "bf16"], gc: Tensor[(48, 1024), "f32"], scratch: Tensor[(4096,), "bf16"], out: Tensor[(1024, 48, 128), "bf16"]
):
    with Mesh((Topology("cta", 96),), layout=(2, 48), names=("vb", "h")) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            st = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 8, 4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        attrs=(S(1), S(3), S(5)),
                        mesh=Mesh((Topology("thread", 384),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 384),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")
            ) as scope_1:
                T.fill(st, 0.0)
            value = T.tensor_view(T.ptr_of(out), layout=(1024, 48, 128), shape=(1024, 48, 128))
            T.fill(value, 0.0)
            value_1_stages = (
                T.tensor_view(49152, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_6_stages = (
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_11_stages = (
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_16_stages = (
                T.tensor_view(172032, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_20_stages = (
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(196608, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_24_stages = (
                T.tensor_view(222720, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
                T.tensor_view(222976, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
            )
            value_28_stages = (
                T.tensor_view(221184, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221440, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221696, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            value_44_stages = (
                T.tensor_view(221952, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222208, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222464, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            value_48_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            value_102_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(155648, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_105_stages = (
                T.tensor_view(204800, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(212992, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            for c in range(0, 16, 1):
                with scope[2:, :32] as scope_2:
                    value_2 = T.tensor_view(
                        T.ptr_of(wup), layout=(1024, 48, 320), shape=(1024, 48, 320)
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(value_2[c * 64:c * 64 + 1, cta.h:cta.h + 1, 0:0 + 1]),
                        layout=((64, 1, 128), (15360, 0, 1)),
                        shape=(64, 1, 128),
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3), layout=((64, 128), (15360, 1)), shape=(64, 128)
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(value_1_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_4, value_5)
                    value_7 = T.tensor_view(
                        T.ptr_of(k), layout=(1024, 16, 128), shape=(1024, 16, 128)
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7[c * 64:c * 64 + 1, cta.h // 3:cta.h // 3 + 1, 0:0 + 1]),
                        layout=((64, 1, 128), (2048, 0, 1)),
                        shape=(64, 1, 128),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_8), layout=((64, 128), (2048, 1)), shape=(64, 128)
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2]),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_9, value_10)
                    value_12 = T.tensor_view(
                        T.ptr_of(q), layout=(1024, 16, 128), shape=(1024, 16, 128)
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12[c * 64:c * 64 + 1, cta.h // 3:cta.h // 3 + 1, 0:0 + 1]),
                        layout=((64, 1, 128), (2048, 0, 1)),
                        shape=(64, 1, 128),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_13), layout=((64, 128), (2048, 1)), shape=(64, 128)
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_11_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_14, value_15)
                    value_17 = T.tensor_view(
                        T.ptr_of(value_2[c * 64:c * 64 + 1, cta.h:cta.h + 1, 128 + (cta.vb * 64):128 + (cta.vb * 64) + 1]),
                        layout=((64, 1, 64), (15360, 0, 1)),
                        shape=(64, 1, 64),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(value_17), layout=((64, 64), (15360, 1)), shape=(64, 64)
                    )
                    value_19 = T.tensor_view(
                        T.ptr_of(value_16_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_18, value_19)
                    value_21 = T.tensor_view(
                        T.ptr_of(value_2[c * 64:c * 64 + 1, cta.h:cta.h + 1, 256:256 + 1]),
                        layout=((64, 1, 64), (15360, 0, 1)),
                        shape=(64, 1, 64),
                    )
                    value_22 = T.tensor_view(
                        T.ptr_of(value_21), layout=((64, 64), (15360, 1)), shape=(64, 64)
                    )
                    value_23 = T.tensor_view(
                        T.ptr_of(value_20_stages[c % 2]),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_22, value_23)
                    value_25 = T.tensor_view(T.ptr_of(gc), layout=(48, 1024), shape=(48, 1024))
                    value_26 = T.tensor_view(
                        T.ptr_of(value_25[cta.h:cta.h + 1, c * 64:c * 64 + 1]),
                        layout=((1, 64), (0, 1)),
                        shape=(1, 64),
                    )
                    value_27 = T.tensor_view(
                        T.ptr_of(value_24_stages[c % 2]), layout=(1, 64), shape=(1, 64)
                    )
                    T.copy_async_tensor(value_26, value_27)
                with Mesh((Topology("thread", 384),), layout=(64,), names=("t",)) as scope_3:
                    value_29 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                    value_30 = T.tensor_view(
                        T.ptr_of(value_24_stages[c % 2]), layout=(1, 64), shape=(1, 64)
                    )
                    value_31 = T.tensor_view(T.ptr_of(value_30), layout=(1, 64), shape=(1, 64))
                    value_32 = T.tensor_view(T.ptr_of(value_31), layout=(1, 64), shape=(1, 64))
                    value_33 = T.tensor_view(T.ptr_of(value_32), layout=(1, 64), shape=(1, 64))
                    value_34 = T.tensor_view(T.ptr_of(value_33), layout=(1, 64), shape=(1, 64))
                    value_35 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 1, 63:63 + 1]),
                        layout=((1, 1), (0, 0)),
                        shape=(1, 1),
                    )
                    value_36 = T.tensor_view(
                        T.ptr_of(value_35), layout=((1,), (0,)), shape=(1,)
                    )
                    value_37 = T.tensor_view(T.ptr_of(value_29), layout=(1,), shape=(1,))
                    T.copy(value_36, value_37)
                    value_38 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.t,), "rmem"]
                    )
                    value_39 = T.tensor_view(T.ptr_of(value_34), layout=(64,), shape=(64,))
                    value_40 = T.tensor_view(
                        T.ptr_of(value_38), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    dst_view = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 64]), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    T.copy(value_39, dst_view)
                    value_41 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.t,), "rmem"]
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 64]), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    T.binary(value_37, rhs_view, value_41, kind=BinaryKind.SUB)
                    value_42 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.t,), "rmem"]
                    )
                    T.unary(value_41, value_42, kind=UnaryKind.EXP)
                    value_43 = T.tensor_view(
                        T.ptr_of(value_28_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    T.copy(value_42, value_43)
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.t,), "rmem"]
                    )
                    src_view = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 64]), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    dst_view_1 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 64]), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    T.unary(src_view, dst_view_1, kind=UnaryKind.EXP)
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.t,), "rmem"]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 0.08838834764831845)
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 64]), layout=(64 @ scope_3.t,), shape=(64,)
                    )
                    T.binary(lhs_view, scalar, value_46, kind=BinaryKind.MUL)
                    value_47 = T.tensor_view(
                        T.ptr_of(value_44_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    T.copy(value_46, value_47)
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_4:
                    value_49 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 64),
                            "bf16",
                            ((2, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 8, 4 @ scope_4.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_50 = T.tensor_view(
                        T.ptr_of(st),
                        layout=((2, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 8, 4 @ scope_4.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_51 = T.tensor_view(
                        T.ptr_of(value_50),
                        layout=((2, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 8, 4 @ scope_4.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_52 = T.tensor_view(
                        T.ptr_of(value_51),
                        layout=((2, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 8, 4 @ scope_4.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    src_view_1 = T.tensor_view(
                        T.ptr_of(value_52[0:0 + 128, 0:0 + 64]),
                        layout=((2, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 8, 4 @ scope_4.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    T.cast(src_view_1, value_49, dtype="bf16")
                    value_53 = T.tensor_view(
                        T.ptr_of(value_48_stages[c % 3]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy(value_49, value_53)
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_5:
                    value_54 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    value_55 = T.tensor_view(
                        T.ptr_of(value_24_stages[c % 2]), layout=(1, 64), shape=(1, 64)
                    )
                    value_56 = T.tensor_view(T.ptr_of(value_55), layout=(1, 64), shape=(1, 64))
                    value_57 = T.tensor_view(T.ptr_of(value_56), layout=(1, 64), shape=(1, 64))
                    value_58 = T.tensor_view(T.ptr_of(value_57), layout=(1, 64), shape=(1, 64))
                    value_59 = T.tensor_view(T.ptr_of(value_58), layout=(1, 64), shape=(1, 64))
                    value_60 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 1, 63:63 + 1]),
                        layout=((1, 1), (0, 0)),
                        shape=(1, 1),
                    )
                    value_61 = T.tensor_view(
                        T.ptr_of(value_60), layout=((1, 1), (0, 0)), shape=(1, 1)
                    )
                    value_62 = T.tensor_view(T.ptr_of(value_54), layout=(1, 1), shape=(1, 1))
                    T.copy(value_61, value_62)
                    el = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    T.unary(value_62, el, kind=UnaryKind.EXP)
                    value_63 = T.tensor_view(
                        T.ptr_of(st),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_64 = T.tensor_view(
                        T.ptr_of(value_63),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(value_64),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_66 = T.tensor_view(
                        T.ptr_of(value_65),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 128, 0:0 + 64]),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    dst_view_2 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 128, 0:0 + 64]),
                        layout=((2, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    T.binary(lhs_view_1, el, dst_view_2, kind=BinaryKind.MUL)
                    value_68 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_69 = T.tensor_view(
                        T.ptr_of(value_16_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_70 = T.tensor_view(
                        T.ptr_of(value_69),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_71 = T.tensor_view(
                        T.ptr_of(value_70),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_72 = T.tensor_view(
                        T.ptr_of(value_71),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_73 = T.tensor_view(
                        T.ptr_of(value_72),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_74 = T.tensor_view(
                        T.ptr_of(value_68),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    dst_view_3 = T.tensor_view(
                        T.ptr_of(value_74[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_73, dst_view_3)
                    uu = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    src_view_2 = T.tensor_view(
                        T.ptr_of(value_74[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.cast(src_view_2, uu, dtype="f32")
                    vn = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(vn, 0.0)
                    acc_view = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    value_75 = T.tensor_view(
                        T.ptr_of(value_1_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_76 = T.tensor_view(
                        T.ptr_of(value_75),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_77 = T.tensor_view(
                        T.ptr_of(value_76),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_78 = T.tensor_view(
                        T.ptr_of(value_77),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_79 = T.tensor_view(
                        T.ptr_of(value_78),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    lhs_view_2 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 0:0 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    value_80 = T.tensor_view(
                        T.ptr_of(value_48_stages[c % 3]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_81 = T.tensor_view(
                        T.ptr_of(value_80),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_82 = T.tensor_view(
                        T.ptr_of(value_81),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_83 = T.tensor_view(
                        T.ptr_of(value_82),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_83[0:0 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view_2,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_3 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 16:16 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_2 = T.tensor_view(
                        T.ptr_of(value_83[16:16 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        lhs_view_3,
                        rhs_view_2,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_2 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_4 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 32:32 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_3 = T.tensor_view(
                        T.ptr_of(value_83[32:32 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_2,
                        lhs_view_4,
                        rhs_view_3,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_3 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_5 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 48:48 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_4 = T.tensor_view(
                        T.ptr_of(value_83[48:48 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_3,
                        lhs_view_5,
                        rhs_view_4,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_4 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_6 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 64:64 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_5 = T.tensor_view(
                        T.ptr_of(value_83[64:64 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_4,
                        lhs_view_6,
                        rhs_view_5,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_5 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_7 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 80:80 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_6 = T.tensor_view(
                        T.ptr_of(value_83[80:80 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_5,
                        lhs_view_7,
                        rhs_view_6,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_6 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_8 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 96:96 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_7 = T.tensor_view(
                        T.ptr_of(value_83[96:96 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_6,
                        lhs_view_8,
                        rhs_view_7,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_7 = T.tensor_view(
                        T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_9 = T.tensor_view(
                        T.ptr_of(value_79[0:0 + 64, 112:112 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_8 = T.tensor_view(
                        T.ptr_of(value_83[112:112 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_7,
                        lhs_view_9,
                        rhs_view_8,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    vn_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_84 = T.tensor_view(
                        T.ptr_of(vn),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    rhs_view_9 = T.tensor_view(
                        T.ptr_of(value_84[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.binary(uu, rhs_view_9, vn_1, kind=BinaryKind.SUB)
                    value_85 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_86 = T.tensor_view(
                        T.ptr_of(value_28_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    value_87 = T.tensor_view(T.ptr_of(value_86), layout=(64,), shape=(64,))
                    value_88 = T.tensor_view(T.ptr_of(value_87), layout=(64,), shape=(64,))
                    value_89 = T.tensor_view(T.ptr_of(value_88), layout=(64,), shape=(64,))
                    value_90 = T.tensor_view(T.ptr_of(value_89), layout=(64,), shape=(64,))
                    value_91 = T.tensor_view(
                        T.ptr_of(value_90), layout=((64, 1), (1, 0)), shape=(64, 1)
                    )
                    value_92 = T.tensor_view(
                        T.ptr_of(value_85),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    dst_view_4 = T.tensor_view(
                        T.ptr_of(value_92[0:0 + 64, 0:0 + 1]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_91, dst_view_4)
                    value_93 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    rhs_view_10 = T.tensor_view(
                        T.ptr_of(value_92[0:0 + 64, 0:0 + 1]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.binary(vn_1, rhs_view_10, value_93, kind=BinaryKind.MUL)
                    value_94 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_93, value_94, dtype="bf16")
                    value_67 = T.tensor_view(
                        223232,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_95 = T.tensor_view(
                        T.ptr_of(value_67),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_94, value_95)
                    acc_view_8 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    value_96 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2]),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_97 = T.tensor_view(
                        T.ptr_of(value_96),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_98 = T.tensor_view(
                        T.ptr_of(value_97),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_99 = T.tensor_view(
                        T.ptr_of(value_98),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_100 = T.tensor_view(
                        T.ptr_of(value_99),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_101 = T.tensor_view(
                        T.ptr_of(value_100),
                        layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    lhs_view_10 = T.tensor_view(
                        T.ptr_of(value_101[0:0 + 64, 0:0 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_11 = T.tensor_view(
                        T.ptr_of(value_95[0:0 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_8,
                        lhs_view_10,
                        rhs_view_11,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_9 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_11 = T.tensor_view(
                        T.ptr_of(value_101[0:0 + 64, 16:16 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_12 = T.tensor_view(
                        T.ptr_of(value_95[16:16 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_9,
                        lhs_view_11,
                        rhs_view_12,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_10 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_12 = T.tensor_view(
                        T.ptr_of(value_101[0:0 + 64, 32:32 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_13 = T.tensor_view(
                        T.ptr_of(value_95[32:32 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_10,
                        lhs_view_12,
                        rhs_view_13,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_11 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_13 = T.tensor_view(
                        T.ptr_of(value_101[0:0 + 64, 48:48 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_14 = T.tensor_view(
                        T.ptr_of(value_95[48:48 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_11,
                        lhs_view_13,
                        rhs_view_14,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_12 = T.tensor_view(
                        T.ptr_of(value_66[64:64 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_14 = T.tensor_view(
                        T.ptr_of(value_101[64:64 + 64, 0:0 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_15 = T.tensor_view(
                        T.ptr_of(value_95[0:0 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_12,
                        lhs_view_14,
                        rhs_view_15,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_13 = T.tensor_view(
                        T.ptr_of(value_66[64:64 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_15 = T.tensor_view(
                        T.ptr_of(value_101[64:64 + 64, 16:16 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_16 = T.tensor_view(
                        T.ptr_of(value_95[16:16 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_13,
                        lhs_view_15,
                        rhs_view_16,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_14 = T.tensor_view(
                        T.ptr_of(value_66[64:64 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_16 = T.tensor_view(
                        T.ptr_of(value_101[64:64 + 64, 32:32 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_17 = T.tensor_view(
                        T.ptr_of(value_95[32:32 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_14,
                        lhs_view_16,
                        rhs_view_17,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_15 = T.tensor_view(
                        T.ptr_of(value_66[64:64 + 64, 0:0 + 64]),
                        layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_17 = T.tensor_view(
                        T.ptr_of(value_101[64:64 + 64, 48:48 + 16]),
                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_18 = T.tensor_view(
                        T.ptr_of(value_95[48:48 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_15,
                        lhs_view_17,
                        rhs_view_18,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                    )
                    value_103 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 8, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(vn_1, value_103, dtype="bf16")
                    value_104 = T.tensor_view(
                        T.ptr_of(value_102_stages[c % 3]),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_103, value_104)
                with Mesh(
                    scope[1:2], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_6:
                    o = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(o, 0.0)
                    acc_view_16 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    value_106 = T.tensor_view(
                        T.ptr_of(value_11_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_107 = T.tensor_view(
                        T.ptr_of(value_106),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_108 = T.tensor_view(
                        T.ptr_of(value_107),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_109 = T.tensor_view(
                        T.ptr_of(value_108),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    value_110 = T.tensor_view(
                        T.ptr_of(value_109),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    lhs_view_18 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 0:0 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    value_111 = T.tensor_view(
                        T.ptr_of(value_48_stages[c % 3]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_112 = T.tensor_view(
                        T.ptr_of(value_111),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_113 = T.tensor_view(
                        T.ptr_of(value_112),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    value_114 = T.tensor_view(
                        T.ptr_of(value_113),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    rhs_view_19 = T.tensor_view(
                        T.ptr_of(value_114[0:0 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_16,
                        lhs_view_18,
                        rhs_view_19,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_17 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_19 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 16:16 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_20 = T.tensor_view(
                        T.ptr_of(value_114[16:16 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_17,
                        lhs_view_19,
                        rhs_view_20,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_18 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_20 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 32:32 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_21 = T.tensor_view(
                        T.ptr_of(value_114[32:32 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_18,
                        lhs_view_20,
                        rhs_view_21,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_19 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_21 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 48:48 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_22 = T.tensor_view(
                        T.ptr_of(value_114[48:48 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_19,
                        lhs_view_21,
                        rhs_view_22,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_20 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_22 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 64:64 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_23 = T.tensor_view(
                        T.ptr_of(value_114[64:64 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_20,
                        lhs_view_22,
                        rhs_view_23,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_21 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_23 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 80:80 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_24 = T.tensor_view(
                        T.ptr_of(value_114[80:80 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_21,
                        lhs_view_23,
                        rhs_view_24,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_22 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_24 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 96:96 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_25 = T.tensor_view(
                        T.ptr_of(value_114[96:96 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_22,
                        lhs_view_24,
                        rhs_view_25,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_23 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_25 = T.tensor_view(
                        T.ptr_of(value_110[0:0 + 64, 112:112 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_26 = T.tensor_view(
                        T.ptr_of(value_114[112:112 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_23,
                        lhs_view_25,
                        rhs_view_26,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    value_115 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 1), (16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_116 = T.tensor_view(
                        T.ptr_of(value_44_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    value_117 = T.tensor_view(T.ptr_of(value_116), layout=(64,), shape=(64,))
                    value_118 = T.tensor_view(T.ptr_of(value_117), layout=(64,), shape=(64,))
                    value_119 = T.tensor_view(T.ptr_of(value_118), layout=(64,), shape=(64,))
                    value_120 = T.tensor_view(T.ptr_of(value_119), layout=(64,), shape=(64,))
                    value_121 = T.tensor_view(
                        T.ptr_of(value_120), layout=((64, 1), (1, 0)), shape=(64, 1)
                    )
                    value_122 = T.tensor_view(
                        T.ptr_of(value_115),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    dst_view_5 = T.tensor_view(
                        T.ptr_of(value_122[0:0 + 64, 0:0 + 1]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_121, dst_view_5)
                    o_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_123 = T.tensor_view(
                        T.ptr_of(o),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    lhs_view_26 = T.tensor_view(
                        T.ptr_of(value_123[0:0 + 64, 0:0 + 64]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    rhs_view_27 = T.tensor_view(
                        T.ptr_of(value_122[0:0 + 64, 0:0 + 1]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_view_26, rhs_view_27, o_1, kind=BinaryKind.MUL)
                    value_124 = T.tensor_view(
                        T.ptr_of(value_20_stages[c % 2]),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_125 = T.tensor_view(
                        T.ptr_of(value_124),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_126 = T.tensor_view(
                        T.ptr_of(value_125),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_127 = T.tensor_view(
                        T.ptr_of(value_126),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_128 = T.tensor_view(
                        T.ptr_of(value_127),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    lhs_view_27 = T.tensor_view(
                        T.ptr_of(value_128[0:0 + 64, 0:0 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    value_129 = T.tensor_view(
                        T.ptr_of(value_102_stages[c % 3]),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_130 = T.tensor_view(
                        T.ptr_of(value_129),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_131 = T.tensor_view(
                        T.ptr_of(value_130),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_132 = T.tensor_view(
                        T.ptr_of(value_131),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_133 = T.tensor_view(
                        T.ptr_of(value_132),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    rhs_view_28 = T.tensor_view(
                        T.ptr_of(value_133[0:0 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        o_1,
                        lhs_view_27,
                        rhs_view_28,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_28 = T.tensor_view(
                        T.ptr_of(value_128[0:0 + 64, 16:16 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_29 = T.tensor_view(
                        T.ptr_of(value_133[16:16 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        o_1,
                        lhs_view_28,
                        rhs_view_29,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_29 = T.tensor_view(
                        T.ptr_of(value_128[0:0 + 64, 32:32 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_30 = T.tensor_view(
                        T.ptr_of(value_133[32:32 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        o_1,
                        lhs_view_29,
                        rhs_view_30,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_30 = T.tensor_view(
                        T.ptr_of(value_128[0:0 + 64, 48:48 + 16]),
                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view_31 = T.tensor_view(
                        T.ptr_of(value_133[48:48 + 16, 0:0 + 64]),
                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        o_1,
                        lhs_view_30,
                        rhs_view_31,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    value_134 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_135 = T.tensor_view(
                        T.ptr_of(o_1),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 8, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.cast(value_135, value_134, dtype="bf16")
                    value_136 = T.tensor_view(
                        T.ptr_of(value_105_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_134, value_136)
                with scope[1:2, :32] as scope_7:
                    value_138 = T.tensor_view(
                        T.ptr_of(value_105_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_139 = T.tensor_view(
                        T.ptr_of(value_138),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_140 = T.tensor_view(
                        T.ptr_of(value_139),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_141 = T.tensor_view(
                        T.ptr_of(value_140),
                        layout=((64, 1, 64), (64, 0, 1)) | Swizzle(3, 4, 3),
                        shape=(64, 1, 64),
                    )
                    value_137_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 4096]), layout=(64, 1, 64), shape=(64, 1, 64)
                    )
                    value_142 = T.tensor_view(
                        T.ptr_of(value_137_1), layout=(64, 1, 64), shape=(64, 1, 64)
                    )
                    T.copy_async_tensor(value_141, value_142)
                    value_143 = T.tensor_view(
                        T.ptr_of(out), layout=(1024, 48, 128), shape=(1024, 48, 128)
                    )
                    value_144 = T.tensor_view(
                        T.ptr_of(value_143), layout=(1024, 48, 128), shape=(1024, 48, 128)
                    )
                    value_145 = T.tensor_view(
                        T.ptr_of(value_144), layout=(1024, 48, 128), shape=(1024, 48, 128)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_145[c * 64:c * 64 + 64, cta.h:cta.h + 1, cta.vb * 64:cta.vb * 64 + 64]),
                        layout=((64, 1, 64), (6144, 128, 1)),
                        shape=(64, 1, 64),
                    )
                    T.copy(value_142, window)
