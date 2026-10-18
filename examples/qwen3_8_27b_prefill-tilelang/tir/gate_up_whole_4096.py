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
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_3_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for g in range(0, 2, 1):
                for bn in range(0, 136, 1):
                    for mi in range(((cta.d0 - (64 * g)) - (16 * bn)) - (132 * (((cta.d0 - (64 * g)) - (16 * bn)) // 132)), 16, 132):
                        acc = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "f32",
                                ShardLayout(
                                    layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    attrs=(S(0), S(1), S(3), S(5)),
                                    mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                                ),
                                "rmem",
                            ]
                        )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_1:
                            T.fill(acc, 0.0)
                        for k in range(0, 5120, 64):
                            with scope[2:, :32] as scope_2:
                                value_1 = T.tensor_view(
                                    T.ptr_of(x[0:0 + 1, ((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                value_2 = T.tensor_view(
                                    T.ptr_of(value_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_1, value_2)
                                value_4 = T.tensor_view(
                                    T.ptr_of(w[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_3_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                T.copy_async_tensor(value_4, value_5)
                            with Mesh(
                                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_3:
                                for o_m in range(0, 128, 128):
                                    for o_n in range(0, 256, 256):
                                        for o_k in range(0, 64, 64):
                                            value_6 = T.tensor_view(
                                                T.ptr_of(acc),
                                                layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                                shape=(128, 256),
                                            )
                                            value_7 = T.tensor_view(
                                                T.ptr_of(value_stages[stage_counter % 3]),
                                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                                shape=(128, 64),
                                            )
                                            lhs_view = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            value_8 = T.tensor_view(
                                                T.ptr_of(value_3_stages[stage_counter % 3]),
                                                layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                                shape=(64, 256),
                                            )
                                            rhs_view = T.tensor_view(
                                                T.ptr_of(value_8[o_k:o_k + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view,
                                                rhs_view,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_1,
                                                rhs_view_1,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_2,
                                                rhs_view_2,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
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
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_4:
                            value_10 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_11 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(value_11, value_10, dtype="bf16")
                            value_9 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_12 = T.tensor_view(
                                T.ptr_of(value_9),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(value_10, value_12)
                        with Mesh(
                            (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                        ) as scope_5:
                            value_14 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            value_9_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_15 = T.tensor_view(
                                T.ptr_of(value_9_2),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_16 = T.tensor_view(
                                T.ptr_of(value_14),
                                layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                shape=(128, 128),
                            )
                            T.copy(value_15, value_16)
                            g32 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.cast(value_16, g32, dtype="f32")
                            value_17 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.unary(g32, value_17, kind=UnaryKind.NEG)
                            value_18 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.unary(value_17, value_18, kind=UnaryKind.EXP)
                            value_19 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                            T.fill(scalar, 1.0)
                            T.binary(scalar, value_18, value_19, kind=BinaryKind.ADD)
                            value_20 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "f32",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.binary(g32, value_19, value_20, kind=BinaryKind.DIV)
                            value_21 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.cast(value_20, value_21, dtype="bf16")
                            value_22 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            value_23 = T.tensor_view(
                                T.ptr_of(value_9_2),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) + 16384 | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_24 = T.tensor_view(
                                T.ptr_of(value_22),
                                layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                shape=(128, 128),
                            )
                            T.copy(value_23, value_24)
                            act = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 128),
                                    "bf16",
                                    (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 8),
                                    "rmem",
                                ]
                            )
                            T.binary(value_21, value_24, act, kind=BinaryKind.MUL)
                            value_13 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_25 = T.tensor_view(
                                T.ptr_of(value_13),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            T.copy(act, value_25)
                        with scope[:1, :32] as scope_6:
                            value_13_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_26 = T.tensor_view(
                                T.ptr_of(value_13_2),
                                layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            value_27 = T.tensor_view(
                                T.ptr_of(out[(2048 * g) + (128 * mi):(2048 * g) + (128 * mi) + 1, 128 * bn:128 * bn + 1]),
                                layout=((128, 128), (17408, 1)),
                                shape=(128, 128),
                            )
                            T.copy_async_tensor(value_26, value_27)
            value_28_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter_1, 0)
            stage_one_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one_1, 1)
            value_31_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for bn_2 in range(0, 0, 1):
                for mi_2 in range(((cta.d0 - (16 * bn_2)) - (132 * (((cta.d0 - (16 * bn_2)) - 64) // 132))) - 64, 16, 132):
                    acc_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 256),
                            "f32",
                            ShardLayout(
                                layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                attrs=(S(0), S(1), S(3), S(5)),
                                mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                            ),
                            "rmem",
                        ]
                    )
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_7:
                        T.fill(acc_1, 0.0)
                    for k_2 in range(0, 5120, 64):
                        with scope[2:, :32] as scope_8:
                            value_29 = T.tensor_view(
                                T.ptr_of(x[0:0 + 1, ((1 * 16) + mi_2) * 128:((1 * 16) + mi_2) * 128 + 1, k_2:k_2 + 1]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            value_30 = T.tensor_view(
                                T.ptr_of(value_28_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value_29, value_30)
                            value_32 = T.tensor_view(
                                T.ptr_of(w[bn_2 * 256:bn_2 * 256 + 1, k_2:k_2 + 1]),
                                layout=((256, 64), (5120, 1)),
                                shape=(256, 64),
                            )
                            value_33 = T.tensor_view(
                                T.ptr_of(value_31_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            T.copy_async_tensor(value_32, value_33)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_9:
                            for o_m_1 in range(0, 128, 128):
                                for o_n_1 in range(0, 256, 256):
                                    for o_k_1 in range(0, 64, 64):
                                        value_34 = T.tensor_view(
                                            T.ptr_of(acc_1),
                                            layout=((2 @ scope_9.d0, 4 @ scope_9.d1, 2, 8 @ scope_9.d2, 32, 4 @ scope_9.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                            shape=(128, 256),
                                        )
                                        value_35 = T.tensor_view(
                                            T.ptr_of(value_28_stages[stage_counter_1 % 3]),
                                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                            shape=(128, 64),
                                        )
                                        lhs_view_4 = T.tensor_view(
                                            T.ptr_of(value_35[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        value_36 = T.tensor_view(
                                            T.ptr_of(value_31_stages[stage_counter_1 % 3]),
                                            layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                            shape=(64, 256),
                                        )
                                        rhs_view_4 = T.tensor_view(
                                            T.ptr_of(value_36[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_34,
                                            lhs_view_4,
                                            rhs_view_4,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_5 = T.tensor_view(
                                            T.ptr_of(value_35[o_m_1:o_m_1 + 128, o_k_1 + 16:o_k_1 + 16 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_5 = T.tensor_view(
                                            T.ptr_of(value_36[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_34,
                                            lhs_view_5,
                                            rhs_view_5,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_6 = T.tensor_view(
                                            T.ptr_of(value_35[o_m_1:o_m_1 + 128, o_k_1 + 32:o_k_1 + 32 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_6 = T.tensor_view(
                                            T.ptr_of(value_36[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_34,
                                            lhs_view_6,
                                            rhs_view_6,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_7 = T.tensor_view(
                                            T.ptr_of(value_35[o_m_1:o_m_1 + 128, o_k_1 + 48:o_k_1 + 48 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_7 = T.tensor_view(
                                            T.ptr_of(value_36[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_34,
                                            lhs_view_7,
                                            rhs_view_7,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                        T.binary(
                            stage_counter_1,
                            stage_one_1,
                            stage_counter_1,
                            kind=BinaryKind.ADD,
                        )
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_10:
                        value_38 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_10.d0, 4 @ scope_10.d1, 2, 8 @ scope_10.d2, 32, 4 @ scope_10.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_39 = T.tensor_view(
                            T.ptr_of(acc_1),
                            layout=((2 @ scope_10.d0, 4 @ scope_10.d1, 2, 8 @ scope_10.d2, 32, 4 @ scope_10.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(value_39, value_38, dtype="bf16")
                        value_37 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_40 = T.tensor_view(
                            T.ptr_of(value_37),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(value_38, value_40)
                    with Mesh(
                        (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                    ) as scope_11:
                        value_42 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        value_37_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_43 = T.tensor_view(
                            T.ptr_of(value_37_2),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_44 = T.tensor_view(
                            T.ptr_of(value_42),
                            layout=(16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                            shape=(128, 128),
                        )
                        T.copy(value_43, value_44)
                        g32_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.cast(value_44, g32_1, dtype="f32")
                        value_45 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.unary(g32_1, value_45, kind=UnaryKind.NEG)
                        value_46 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.unary(value_45, value_46, kind=UnaryKind.EXP)
                        value_47 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_1, 1.0)
                        T.binary(scalar_1, value_46, value_47, kind=BinaryKind.ADD)
                        value_48 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "f32",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.binary(g32_1, value_47, value_48, kind=BinaryKind.DIV)
                        value_49 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.cast(value_48, value_49, dtype="bf16")
                        value_50 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        value_51 = T.tensor_view(
                            T.ptr_of(value_37_2),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) + 16384 | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_52 = T.tensor_view(
                            T.ptr_of(value_50),
                            layout=(16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                            shape=(128, 128),
                        )
                        T.copy(value_51, value_52)
                        act_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 128),
                                "bf16",
                                (16 @ scope_11.d0, 8, 16 @ scope_11.d1, 8),
                                "rmem",
                            ]
                        )
                        T.binary(value_49, value_52, act_1, kind=BinaryKind.MUL)
                        value_41 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_53 = T.tensor_view(
                            T.ptr_of(value_41),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        T.copy(act_1, value_53)
                    with scope[:1, :32] as scope_12:
                        value_41_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_55 = T.tensor_view(
                            T.ptr_of(value_41_2),
                            layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        value_54_1 = T.tensor_view(
                            T.ptr_of(scratch[0:0 + 16384]), layout=(128, 128), shape=(128, 128)
                        )
                        value_56 = T.tensor_view(
                            T.ptr_of(value_54_1), layout=(128, 128), shape=(128, 128)
                        )
                        T.copy_async_tensor(value_55, value_56)
                        value_57 = T.tensor_view(
                            T.ptr_of(out), layout=(4096, 17408), shape=(4096, 17408)
                        )
                        window = T.tensor_view(
                            T.ptr_of(value_57[((1 * 16) + mi_2) * 128:((1 * 16) + mi_2) * 128 + 128, bn_2 * 128:bn_2 * 128 + 128]),
                            layout=((128, 128), (17408, 1)),
                            shape=(128, 128),
                        )
                        T.copy(value_56, window)
