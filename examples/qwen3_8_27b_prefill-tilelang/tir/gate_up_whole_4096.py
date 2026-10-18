# analysis target=nvidia.h200_sxm module=GateUp function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r354.78MB/w153.06MB@logical,r133.07GB/w19.73GB@total,r1.01GB/w153.06MB@cta,r1.01GB/w153.06MB@thread;rmem:r2.85GB/w2.80GB@logical,r376.20GB/w373.91GB@total,r2.85GB/w2.83GB@cta,r12.10MB/w11.33MB@thread;smem:r1.02GB/w371.84MB@logical,r135.27GB/w135.27GB@total,r1.02GB/w1.02GB@cta,r184.55MB/w1023.85MB@thread footprint=v47:307:32.00KB;v50:308:4.12MB;v93:330:0;v96:331:0;w:288.00KB;x:256.00KB footprint-precision=lower_bound peak=gmem:516.03MB;rmem:192.00KB;smem:208.00KB persistent=gmem:380.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 4096, 5120), "bf16"], w: Tensor[(34816, 5120), "bf16"], scratch: Tensor[(16384,), "bf16"], out: Tensor[(4096, 17408), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(4096, 17408), shape=(4096, 17408))
            T.fill(value, 0.0)
            value_1_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_7_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for g in range(0, 2, 1):
                for bn in range(0, 136, 1):
                    for mi in range(((cta.persistent - (64 * g)) - (16 * bn)) - (132 * (((cta.persistent - (64 * g)) - (16 * bn)) // 132)), 16, 132):
                        acc = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "f32",
                                ShardLayout(
                                    layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    attrs=(S(0), S(1), S(3), S(5)),
                                    mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
                                ),
                                "rmem",
                            ]
                        )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_1:
                            T.fill(acc, 0.0)
                        for k in range(0, 5120, 64):
                            with scope[2:, :32] as scope_2:
                                value_2 = T.tensor_view(
                                    T.ptr_of(x), layout=(1, 4096, 5120), shape=(1, 4096, 5120)
                                )
                                value_3 = T.tensor_view(
                                    T.ptr_of(value_2), layout=(4096, 5120), shape=(4096, 5120)
                                )
                                value_4 = T.tensor_view(
                                    T.ptr_of(value_3), layout=(4096, 5120), shape=(4096, 5120)
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_4[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(value_1_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_5, value_6)
                                value_8 = T.tensor_view(
                                    T.ptr_of(w), layout=(34816, 5120), shape=(34816, 5120)
                                )
                                value_9 = T.tensor_view(
                                    T.ptr_of(value_8[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                value_10 = T.tensor_view(
                                    T.ptr_of(value_7_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                T.copy_async_tensor(value_9, value_10)
                            with Mesh(
                                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                            ) as scope_3:
                                value_11 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_12 = T.tensor_view(
                                    T.ptr_of(value_11),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_13 = T.tensor_view(
                                    T.ptr_of(value_12),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                acc_view = T.tensor_view(
                                    T.ptr_of(value_13[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_14 = T.tensor_view(
                                    T.ptr_of(value_1_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                value_15 = T.tensor_view(
                                    T.ptr_of(value_14),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                value_16 = T.tensor_view(
                                    T.ptr_of(value_15),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                value_17 = T.tensor_view(
                                    T.ptr_of(value_16),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                value_18 = T.tensor_view(
                                    T.ptr_of(value_17),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_18[0:0 + 128, 0:0 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_19 = T.tensor_view(
                                    T.ptr_of(value_7_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_20 = T.tensor_view(
                                    T.ptr_of(value_19),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_21 = T.tensor_view(
                                    T.ptr_of(value_20),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_22 = T.tensor_view(
                                    T.ptr_of(value_21),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_23 = T.tensor_view(
                                    T.ptr_of(value_22),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_24 = T.tensor_view(
                                    T.ptr_of(value_23),
                                    layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_24[0:0 + 16, 0:0 + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                acc_view_1 = T.tensor_view(
                                    T.ptr_of(value_13[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_18[0:0 + 128, 16:16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_24[16:16 + 16, 0:0 + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_1,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                acc_view_2 = T.tensor_view(
                                    T.ptr_of(value_13[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_18[0:0 + 128, 32:32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_24[32:32 + 16, 0:0 + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_2,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                acc_view_3 = T.tensor_view(
                                    T.ptr_of(value_13[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_18[0:0 + 128, 48:48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_24[48:48 + 16, 0:0 + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    acc_view_3,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                            T.binary(
                                stage_counter,
                                stage_one,
                                stage_counter,
                                kind=BinaryKind.ADD,
                            )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_4:
                            value_26 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_27 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_28 = T.tensor_view(
                                T.ptr_of(value_27),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_29 = T.tensor_view(
                                T.ptr_of(value_28),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_30 = T.tensor_view(
                                T.ptr_of(value_29),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            src_view = T.tensor_view(
                                T.ptr_of(value_30[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(src_view, value_26, dtype="bf16")
                            value_25 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_31 = T.tensor_view(
                                T.ptr_of(value_25),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(value_26, value_31)
                        with Mesh(
                            (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                        ) as scope_5:
                            value_33 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            value_25_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_34 = T.tensor_view(
                                T.ptr_of(value_25_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_35 = T.tensor_view(
                                T.ptr_of(value_34),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_36 = T.tensor_view(
                                T.ptr_of(value_35),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_37 = T.tensor_view(
                                T.ptr_of(value_36),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_38 = T.tensor_view(
                                T.ptr_of(value_33),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            dst_view = T.tensor_view(
                                T.ptr_of(value_38[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            T.copy(value_37, dst_view)
                            g32 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            src_view_1 = T.tensor_view(
                                T.ptr_of(value_38[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            T.cast(src_view_1, g32, dtype="f32")
                            value_39 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            T.unary(g32, value_39, kind=UnaryKind.NEG)
                            value_40 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            T.unary(value_39, value_40, kind=UnaryKind.EXP)
                            value_41 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                            T.fill(scalar, 1.0)
                            T.binary(scalar, value_40, value_41, kind=BinaryKind.ADD)
                            value_42 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            T.binary(g32, value_41, value_42, kind=BinaryKind.DIV)
                            value_43 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            T.cast(value_42, value_43, dtype="bf16")
                            value_44 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            value_45 = T.tensor_view(
                                T.ptr_of(value_36),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) + 16384 | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_46 = T.tensor_view(
                                T.ptr_of(value_44),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            dst_view_1 = T.tensor_view(
                                T.ptr_of(value_46[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            T.copy(value_45, dst_view_1)
                            act = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                    "rmem",
                                ]
                            )
                            rhs_view_4 = T.tensor_view(
                                T.ptr_of(value_46[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 8),
                                shape=(128, 128),
                            )
                            T.binary(value_43, rhs_view_4, act, kind=BinaryKind.MUL)
                            value_32 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_47 = T.tensor_view(
                                T.ptr_of(value_32),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            T.copy(act, value_47)
                        with scope[:1, :32] as scope_6:
                            value_32_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_48 = T.tensor_view(
                                T.ptr_of(value_32_2),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_49 = T.tensor_view(
                                T.ptr_of(value_48),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_50 = T.tensor_view(
                                T.ptr_of(value_49),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_51 = T.tensor_view(
                                T.ptr_of(out), layout=(4096, 17408), shape=(4096, 17408)
                            )
                            value_52 = T.tensor_view(
                                T.ptr_of(value_51), layout=(4096, 17408), shape=(4096, 17408)
                            )
                            value_53 = T.tensor_view(
                                T.ptr_of(value_52), layout=(4096, 17408), shape=(4096, 17408)
                            )
                            value_54 = T.tensor_view(
                                T.ptr_of(value_53), layout=(4096, 17408), shape=(4096, 17408)
                            )
                            value_55 = T.tensor_view(
                                T.ptr_of(value_54), layout=(4096, 17408), shape=(4096, 17408)
                            )
                            value_56 = T.tensor_view(
                                T.ptr_of(value_55[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, bn * 128:bn * 128 + 1]),
                                layout=((128, 128), (17408, 1)),
                                shape=(128, 128),
                            )
                            value_57 = T.tensor_view(
                                T.ptr_of(value_56),
                                layout=((128, 128), (17408, 1)),
                                shape=(128, 128),
                            )
                            T.copy_async_tensor(value_50, value_57)
            value_58_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter_1, 0)
            stage_one_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one_1, 1)
            value_64_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for bn_2 in range(0, 0, 1):
                for mi_2 in range(((cta.persistent - (16 * bn_2)) - (132 * (((cta.persistent - (16 * bn_2)) - 64) // 132))) - 64, 16, 132):
                    acc_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 256),
                            "f32",
                            ShardLayout(
                                layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                attrs=(S(0), S(1), S(3), S(5)),
                                mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
                            ),
                            "rmem",
                        ]
                    )
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                    ) as scope_7:
                        T.fill(acc_1, 0.0)
                    for k_2 in range(0, 5120, 64):
                        with scope[2:, :32] as scope_8:
                            value_59 = T.tensor_view(
                                T.ptr_of(x), layout=(1, 4096, 5120), shape=(1, 4096, 5120)
                            )
                            value_60 = T.tensor_view(
                                T.ptr_of(value_59), layout=(4096, 5120), shape=(4096, 5120)
                            )
                            value_61 = T.tensor_view(
                                T.ptr_of(value_60), layout=(4096, 5120), shape=(4096, 5120)
                            )
                            value_62 = T.tensor_view(
                                T.ptr_of(value_61[((1 * 16) + mi_2) * 128:((1 * 16) + mi_2) * 128 + 1, k_2:k_2 + 1]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            value_63 = T.tensor_view(
                                T.ptr_of(value_58_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value_62, value_63)
                            value_65 = T.tensor_view(
                                T.ptr_of(w), layout=(34816, 5120), shape=(34816, 5120)
                            )
                            value_66 = T.tensor_view(
                                T.ptr_of(value_65[bn_2 * 256:bn_2 * 256 + 1, k_2:k_2 + 1]),
                                layout=((256, 64), (5120, 1)),
                                shape=(256, 64),
                            )
                            value_67 = T.tensor_view(
                                T.ptr_of(value_64_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            T.copy_async_tensor(value_66, value_67)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_9:
                            value_68 = T.tensor_view(
                                T.ptr_of(acc_1),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_69 = T.tensor_view(
                                T.ptr_of(value_68),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_70 = T.tensor_view(
                                T.ptr_of(value_69),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            acc_view_4 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_71 = T.tensor_view(
                                T.ptr_of(value_58_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_72 = T.tensor_view(
                                T.ptr_of(value_71),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_73 = T.tensor_view(
                                T.ptr_of(value_72),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_74 = T.tensor_view(
                                T.ptr_of(value_73),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_75 = T.tensor_view(
                                T.ptr_of(value_74),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            lhs_view_4 = T.tensor_view(
                                T.ptr_of(value_75[0:0 + 128, 0:0 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            value_76 = T.tensor_view(
                                T.ptr_of(value_64_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_77 = T.tensor_view(
                                T.ptr_of(value_76),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_78 = T.tensor_view(
                                T.ptr_of(value_77),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_79 = T.tensor_view(
                                T.ptr_of(value_78),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_80 = T.tensor_view(
                                T.ptr_of(value_79),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_81 = T.tensor_view(
                                T.ptr_of(value_80),
                                layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            rhs_view_5 = T.tensor_view(
                                T.ptr_of(value_81[0:0 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_4,
                                lhs_view_4,
                                rhs_view_5,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_5 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_5 = T.tensor_view(
                                T.ptr_of(value_75[0:0 + 128, 16:16 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_6 = T.tensor_view(
                                T.ptr_of(value_81[16:16 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_5,
                                lhs_view_5,
                                rhs_view_6,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_6 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_6 = T.tensor_view(
                                T.ptr_of(value_75[0:0 + 128, 32:32 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_7 = T.tensor_view(
                                T.ptr_of(value_81[32:32 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_6,
                                lhs_view_6,
                                rhs_view_7,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_7 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_7 = T.tensor_view(
                                T.ptr_of(value_75[0:0 + 128, 48:48 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_8 = T.tensor_view(
                                T.ptr_of(value_81[48:48 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_7,
                                lhs_view_7,
                                rhs_view_8,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                        T.binary(
                            stage_counter_1,
                            stage_one_1,
                            stage_counter_1,
                            kind=BinaryKind.ADD,
                        )
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                    ) as scope_10:
                        value_83 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_84 = T.tensor_view(
                            T.ptr_of(acc_1),
                            layout=((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_85 = T.tensor_view(
                            T.ptr_of(value_84),
                            layout=((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_86 = T.tensor_view(
                            T.ptr_of(value_85),
                            layout=((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_87 = T.tensor_view(
                            T.ptr_of(value_86),
                            layout=((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        src_view_2 = T.tensor_view(
                            T.ptr_of(value_87[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_10.group, 4 @ scope_10.warp, 2, 8 @ scope_10.lane8, 32, 4 @ scope_10.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(src_view_2, value_83, dtype="bf16")
                        value_82 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_88 = T.tensor_view(
                            T.ptr_of(value_82),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(value_83, value_88)
                    with Mesh(
                        (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                    ) as scope_11:
                        value_90 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        value_82_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_91 = T.tensor_view(
                            T.ptr_of(value_82_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_92 = T.tensor_view(
                            T.ptr_of(value_91),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_93 = T.tensor_view(
                            T.ptr_of(value_92),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_94 = T.tensor_view(
                            T.ptr_of(value_93),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_95 = T.tensor_view(
                            T.ptr_of(value_90),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        dst_view_2 = T.tensor_view(
                            T.ptr_of(value_95[0:0 + 128, 0:0 + 128]),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        T.copy(value_94, dst_view_2)
                        g32_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        src_view_3 = T.tensor_view(
                            T.ptr_of(value_95[0:0 + 128, 0:0 + 128]),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        T.cast(src_view_3, g32_1, dtype="f32")
                        value_96 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        T.unary(g32_1, value_96, kind=UnaryKind.NEG)
                        value_97 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        T.unary(value_96, value_97, kind=UnaryKind.EXP)
                        value_98 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_1, 1.0)
                        T.binary(scalar_1, value_97, value_98, kind=BinaryKind.ADD)
                        value_99 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        T.binary(g32_1, value_98, value_99, kind=BinaryKind.DIV)
                        value_100 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        T.cast(value_99, value_100, dtype="bf16")
                        value_101 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        value_102 = T.tensor_view(
                            T.ptr_of(value_93),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) + 16384 | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_103 = T.tensor_view(
                            T.ptr_of(value_101),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        dst_view_3 = T.tensor_view(
                            T.ptr_of(value_103[0:0 + 128, 0:0 + 128]),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        T.copy(value_102, dst_view_3)
                        act_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                                "rmem",
                            ]
                        )
                        rhs_view_9 = T.tensor_view(
                            T.ptr_of(value_103[0:0 + 128, 0:0 + 128]),
                            layout=(16 @ scope_11.r, 8, 16 @ scope_11.c, 8),
                            shape=(128, 128),
                        )
                        T.binary(value_100, rhs_view_9, act_1, kind=BinaryKind.MUL)
                        value_89 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_104 = T.tensor_view(
                            T.ptr_of(value_89),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        T.copy(act_1, value_104)
                    with scope[:1, :32] as scope_12:
                        value_89_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_106 = T.tensor_view(
                            T.ptr_of(value_89_2),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_107 = T.tensor_view(
                            T.ptr_of(value_106),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_108 = T.tensor_view(
                            T.ptr_of(value_107),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_105_1 = T.tensor_view(
                            T.ptr_of(scratch[0:0 + 16384]), layout=(128, 128), shape=(128, 128)
                        )
                        value_109 = T.tensor_view(
                            T.ptr_of(value_105_1), layout=(128, 128), shape=(128, 128)
                        )
                        T.copy_async_tensor(value_108, value_109)
                        value_110 = T.tensor_view(
                            T.ptr_of(out), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        value_111 = T.tensor_view(
                            T.ptr_of(value_110), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        value_112 = T.tensor_view(
                            T.ptr_of(value_111), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        value_113 = T.tensor_view(
                            T.ptr_of(value_112), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        value_114 = T.tensor_view(
                            T.ptr_of(value_113), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        value_115 = T.tensor_view(
                            T.ptr_of(value_114), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        window = T.tensor_view(
                            T.ptr_of(value_115[((1 * 16) + mi_2) * 128:((1 * 16) + mi_2) * 128 + 128, bn_2 * 128:bn_2 * 128 + 128]),
                            layout=((128, 128), (17408, 1)),
                            shape=(128, 128),
                        )
                        T.copy(value_109, window)
