# analysis target=nvidia.h200_sxm module=Residual function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r319.00MB/w14.50MB@logical,r59.75GB/w1.87GB@total,r463.50MB/w14.50MB@cta,r463.50MB/w14.50MB@thread;rmem:r1.21GB/w1.20GB@logical,r159.27GB/w159.23GB@total,r1.21GB/w1.21GB@cta,r5.13MB/w4.83MB@thread;smem:r463.50MB/w319.00MB@logical,r59.75GB/w59.75GB@total,r463.50MB/w463.50MB@cta,r81.15MB/w461.26MB@thread footprint=hidden:8.00MB;v39:157:0;v41:158:0;v73:178:64.00KB;v75:179:8.00MB;w:512.00KB;x:128.00KB footprint-precision=lower_bound peak=gmem:224.06MB;rmem:256.00KB;smem:208.00KB persistent=gmem:214.00MB
#   buffer=x holds=186.06MB time=bn space=none reuse=510.00MB fits=no precision=exact
#   error="l2 reuse window bn holds 186.06MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 17408), "bf16"], hidden: Tensor[(1, 1024, 5120), "bf16"], w: ConstTensor[(5120, 17408), "bf16"], scratch: Tensor[(32768,), "bf16"], scratch_1: Tensor[(32768,), "bf16"], out: Tensor[(1024, 5120), "bf16"]
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
            for g in range(0, 0, 1):
                for bn in range(0, 20, 1):
                    for mi in range(((cta.d0 - (28 * g)) - (8 * bn)) - (132 * (((cta.d0 - (28 * g)) - (8 * bn)) // 132)), 8, 132):
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
                        for k in range(0, 17408, 64):
                            with scope[2:, :32] as scope_2:
                                value_1 = T.tensor_view(
                                    T.ptr_of(x[0:0 + 1, ((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (17408, 1)),
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
                                    layout=((256, 64), (17408, 1)),
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
                        with scope[:1, :32] as scope_4:
                            value_10 = T.tensor_view(
                                T.ptr_of(hidden[0:0 + 1, ((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, bn * 256:bn * 256 + 1]),
                                layout=((128, 256), (5120, 1)),
                                shape=(128, 256),
                            )
                            value_9 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_11 = T.tensor_view(
                                T.ptr_of(value_9),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(value_10, value_11)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_5:
                            value_13 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
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
                            value_14 = T.tensor_view(
                                T.ptr_of(value_9_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_15 = T.tensor_view(
                                T.ptr_of(value_13),
                                layout=((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.copy(value_14, value_15)
                            value_16 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_17 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(value_17, value_16, dtype="bf16")
                            y = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            T.binary(value_15, value_16, y, kind=BinaryKind.ADD)
                            value_12 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_18 = T.tensor_view(
                                T.ptr_of(value_12),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(y, value_18)
                        with scope[:1, :32] as scope_6:
                            value_12_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_20 = T.tensor_view(
                                T.ptr_of(value_12_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_19_1 = T.tensor_view(
                                T.ptr_of(scratch[0:0 + 32768]),
                                layout=(128, 256),
                                shape=(128, 256),
                            )
                            value_21 = T.tensor_view(
                                T.ptr_of(value_19_1), layout=(128, 256), shape=(128, 256)
                            )
                            T.copy_async_tensor(value_20, value_21)
                            value_22 = T.tensor_view(
                                T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            window = T.tensor_view(
                                T.ptr_of(value_22[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                                layout=((128, 256), (5120, 1)),
                                shape=(128, 256),
                            )
                            T.copy(value_21, window)
            value_23_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter_1, 0)
            stage_one_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one_1, 1)
            value_26_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for bn_2 in range(0, 16, 1):
                for mi_2 in range((cta.d0 - (8 * bn_2)) - (132 * ((cta.d0 - (8 * bn_2)) // 132)), 8, 132):
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
                    for k_2 in range(0, 17408, 64):
                        with scope[2:, :32] as scope_8:
                            value_24 = T.tensor_view(
                                T.ptr_of(x[0:0 + 1, ((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 1, k_2:k_2 + 1]),
                                layout=((128, 64), (17408, 1)),
                                shape=(128, 64),
                            )
                            value_25 = T.tensor_view(
                                T.ptr_of(value_23_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value_24, value_25)
                            value_27 = T.tensor_view(
                                T.ptr_of(w[bn_2 * 256:bn_2 * 256 + 1, k_2:k_2 + 1]),
                                layout=((256, 64), (17408, 1)),
                                shape=(256, 64),
                            )
                            value_28 = T.tensor_view(
                                T.ptr_of(value_26_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            T.copy_async_tensor(value_27, value_28)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_9:
                            for o_m_1 in range(0, 128, 128):
                                for o_n_1 in range(0, 256, 256):
                                    for o_k_1 in range(0, 64, 64):
                                        value_29 = T.tensor_view(
                                            T.ptr_of(acc_1),
                                            layout=((2 @ scope_9.d0, 4 @ scope_9.d1, 2, 8 @ scope_9.d2, 32, 4 @ scope_9.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                            shape=(128, 256),
                                        )
                                        value_30 = T.tensor_view(
                                            T.ptr_of(value_23_stages[stage_counter_1 % 3]),
                                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                            shape=(128, 64),
                                        )
                                        lhs_view_4 = T.tensor_view(
                                            T.ptr_of(value_30[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        value_31 = T.tensor_view(
                                            T.ptr_of(value_26_stages[stage_counter_1 % 3]),
                                            layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                            shape=(64, 256),
                                        )
                                        rhs_view_4 = T.tensor_view(
                                            T.ptr_of(value_31[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_29,
                                            lhs_view_4,
                                            rhs_view_4,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_5 = T.tensor_view(
                                            T.ptr_of(value_30[o_m_1:o_m_1 + 128, o_k_1 + 16:o_k_1 + 16 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_5 = T.tensor_view(
                                            T.ptr_of(value_31[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_29,
                                            lhs_view_5,
                                            rhs_view_5,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_6 = T.tensor_view(
                                            T.ptr_of(value_30[o_m_1:o_m_1 + 128, o_k_1 + 32:o_k_1 + 32 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_6 = T.tensor_view(
                                            T.ptr_of(value_31[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_29,
                                            lhs_view_6,
                                            rhs_view_6,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_7 = T.tensor_view(
                                            T.ptr_of(value_30[o_m_1:o_m_1 + 128, o_k_1 + 48:o_k_1 + 48 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_7 = T.tensor_view(
                                            T.ptr_of(value_31[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_29,
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
                    with scope[:1, :32] as scope_10:
                        value_33 = T.tensor_view(
                            T.ptr_of(hidden[0:0 + 1, ((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 1, bn_2 * 256:bn_2 * 256 + 1]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        value_32 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_34 = T.tensor_view(
                            T.ptr_of(value_32),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy_async_tensor(value_33, value_34)
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_11:
                        value_36 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.d0, 4 @ scope_11.d1, 2, 8 @ scope_11.d2, 32, 4 @ scope_11.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_32_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_37 = T.tensor_view(
                            T.ptr_of(value_32_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_38 = T.tensor_view(
                            T.ptr_of(value_36),
                            layout=((2 @ scope_11.d0, 4 @ scope_11.d1, 2, 8 @ scope_11.d2, 32, 4 @ scope_11.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.copy(value_37, value_38)
                        value_39 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.d0, 4 @ scope_11.d1, 2, 8 @ scope_11.d2, 32, 4 @ scope_11.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_40 = T.tensor_view(
                            T.ptr_of(acc_1),
                            layout=((2 @ scope_11.d0, 4 @ scope_11.d1, 2, 8 @ scope_11.d2, 32, 4 @ scope_11.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(value_40, value_39, dtype="bf16")
                        y_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.d0, 4 @ scope_11.d1, 2, 8 @ scope_11.d2, 32, 4 @ scope_11.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        T.binary(value_38, value_39, y_1, kind=BinaryKind.ADD)
                        value_35 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_41 = T.tensor_view(
                            T.ptr_of(value_35),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(y_1, value_41)
                    with scope[:1, :32] as scope_12:
                        value_35_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_43 = T.tensor_view(
                            T.ptr_of(value_35_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_42_1 = T.tensor_view(
                            T.ptr_of(scratch_1[0:0 + 32768]),
                            layout=(128, 256),
                            shape=(128, 256),
                        )
                        value_44 = T.tensor_view(
                            T.ptr_of(value_42_1), layout=(128, 256), shape=(128, 256)
                        )
                        T.copy_async_tensor(value_43, value_44)
                        value_45 = T.tensor_view(
                            T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        window_1 = T.tensor_view(
                            T.ptr_of(value_45[((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 128, bn_2 * 256:bn_2 * 256 + 256]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        T.copy(value_44, window_1)
