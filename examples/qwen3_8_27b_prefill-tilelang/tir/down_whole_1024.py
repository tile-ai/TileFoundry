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
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120))
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
            for g in range(0, 0, 1):
                for bn in range(0, 20, 1):
                    for mi in range(((cta.persistent - (28 * g)) - (8 * bn)) - (132 * (((cta.persistent - (28 * g)) - (8 * bn)) // 132)), 8, 132):
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
                        for k in range(0, 17408, 64):
                            with scope[2:, :32] as scope_2:
                                value_2 = T.tensor_view(
                                    T.ptr_of(x), layout=(1, 1024, 17408), shape=(1, 1024, 17408)
                                )
                                value_3 = T.tensor_view(
                                    T.ptr_of(value_2), layout=(1024, 17408), shape=(1024, 17408)
                                )
                                value_4 = T.tensor_view(
                                    T.ptr_of(value_3), layout=(1024, 17408), shape=(1024, 17408)
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_4[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (17408, 1)),
                                    shape=(128, 64),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(value_1_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_5, value_6)
                                value_8 = T.tensor_view(
                                    T.ptr_of(w), layout=(5120, 17408), shape=(5120, 17408)
                                )
                                value_9 = T.tensor_view(
                                    T.ptr_of(value_8[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (17408, 1)),
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
                        with scope[:1, :32] as scope_4:
                            value_26 = T.tensor_view(
                                T.ptr_of(hidden), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
                            )
                            value_27 = T.tensor_view(
                                T.ptr_of(value_26), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_28 = T.tensor_view(
                                T.ptr_of(value_27), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_29 = T.tensor_view(
                                T.ptr_of(value_28[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, bn * 256:bn * 256 + 1]),
                                layout=((128, 256), (5120, 1)),
                                shape=(128, 256),
                            )
                            value_25 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_30 = T.tensor_view(
                                T.ptr_of(value_25),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(value_29, value_30)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_5:
                            value_32 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
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
                            value_33 = T.tensor_view(
                                T.ptr_of(value_25_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_34 = T.tensor_view(
                                T.ptr_of(value_33),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_35 = T.tensor_view(
                                T.ptr_of(value_34),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_36 = T.tensor_view(
                                T.ptr_of(value_32),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            dst_view = T.tensor_view(
                                T.ptr_of(value_36[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.copy(value_35, dst_view)
                            value_37 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_38 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_39 = T.tensor_view(
                                T.ptr_of(value_38),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_40 = T.tensor_view(
                                T.ptr_of(value_39),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_41 = T.tensor_view(
                                T.ptr_of(value_40),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            src_view = T.tensor_view(
                                T.ptr_of(value_41[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(src_view, value_37, dtype="bf16")
                            y = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            lhs_view_4 = T.tensor_view(
                                T.ptr_of(value_36[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.binary(lhs_view_4, value_37, y, kind=BinaryKind.ADD)
                            value_31 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_42 = T.tensor_view(
                                T.ptr_of(value_31),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(y, value_42)
                        with scope[:1, :32] as scope_6:
                            value_31_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_44 = T.tensor_view(
                                T.ptr_of(value_31_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_45 = T.tensor_view(
                                T.ptr_of(value_44),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_46 = T.tensor_view(
                                T.ptr_of(value_45),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_43_1 = T.tensor_view(
                                T.ptr_of(scratch[0:0 + 32768]),
                                layout=(128, 256),
                                shape=(128, 256),
                            )
                            value_47 = T.tensor_view(
                                T.ptr_of(value_43_1), layout=(128, 256), shape=(128, 256)
                            )
                            T.copy_async_tensor(value_46, value_47)
                            value_48 = T.tensor_view(
                                T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_49 = T.tensor_view(
                                T.ptr_of(value_48), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_50 = T.tensor_view(
                                T.ptr_of(value_49), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_51 = T.tensor_view(
                                T.ptr_of(value_50), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            value_52 = T.tensor_view(
                                T.ptr_of(value_51), layout=(1024, 5120), shape=(1024, 5120)
                            )
                            window = T.tensor_view(
                                T.ptr_of(value_52[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                                layout=((128, 256), (5120, 1)),
                                shape=(128, 256),
                            )
                            T.copy(value_47, window)
            value_53_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter_1, 0)
            stage_one_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one_1, 1)
            value_59_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for bn_2 in range(0, 16, 1):
                for mi_2 in range((cta.persistent - (8 * bn_2)) - (132 * ((cta.persistent - (8 * bn_2)) // 132)), 8, 132):
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
                    for k_2 in range(0, 17408, 64):
                        with scope[2:, :32] as scope_8:
                            value_54 = T.tensor_view(
                                T.ptr_of(x), layout=(1, 1024, 17408), shape=(1, 1024, 17408)
                            )
                            value_55 = T.tensor_view(
                                T.ptr_of(value_54), layout=(1024, 17408), shape=(1024, 17408)
                            )
                            value_56 = T.tensor_view(
                                T.ptr_of(value_55), layout=(1024, 17408), shape=(1024, 17408)
                            )
                            value_57 = T.tensor_view(
                                T.ptr_of(value_56[((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 1, k_2:k_2 + 1]),
                                layout=((128, 64), (17408, 1)),
                                shape=(128, 64),
                            )
                            value_58 = T.tensor_view(
                                T.ptr_of(value_53_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value_57, value_58)
                            value_60 = T.tensor_view(
                                T.ptr_of(w), layout=(5120, 17408), shape=(5120, 17408)
                            )
                            value_61 = T.tensor_view(
                                T.ptr_of(value_60[bn_2 * 256:bn_2 * 256 + 1, k_2:k_2 + 1]),
                                layout=((256, 64), (17408, 1)),
                                shape=(256, 64),
                            )
                            value_62 = T.tensor_view(
                                T.ptr_of(value_59_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            T.copy_async_tensor(value_61, value_62)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_9:
                            value_63 = T.tensor_view(
                                T.ptr_of(acc_1),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_64 = T.tensor_view(
                                T.ptr_of(value_63),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_65 = T.tensor_view(
                                T.ptr_of(value_64),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            acc_view_4 = T.tensor_view(
                                T.ptr_of(value_65[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_66 = T.tensor_view(
                                T.ptr_of(value_53_stages[stage_counter_1 % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_67 = T.tensor_view(
                                T.ptr_of(value_66),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_68 = T.tensor_view(
                                T.ptr_of(value_67),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_69 = T.tensor_view(
                                T.ptr_of(value_68),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            value_70 = T.tensor_view(
                                T.ptr_of(value_69),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            lhs_view_5 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 0:0 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            value_71 = T.tensor_view(
                                T.ptr_of(value_59_stages[stage_counter_1 % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_72 = T.tensor_view(
                                T.ptr_of(value_71),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_73 = T.tensor_view(
                                T.ptr_of(value_72),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_74 = T.tensor_view(
                                T.ptr_of(value_73),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_75 = T.tensor_view(
                                T.ptr_of(value_74),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            value_76 = T.tensor_view(
                                T.ptr_of(value_75),
                                layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            rhs_view_4 = T.tensor_view(
                                T.ptr_of(value_76[0:0 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_4,
                                lhs_view_5,
                                rhs_view_4,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_5 = T.tensor_view(
                                T.ptr_of(value_65[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_6 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 16:16 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_5 = T.tensor_view(
                                T.ptr_of(value_76[16:16 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_5,
                                lhs_view_6,
                                rhs_view_5,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_6 = T.tensor_view(
                                T.ptr_of(value_65[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_7 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 32:32 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_6 = T.tensor_view(
                                T.ptr_of(value_76[32:32 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_6,
                                lhs_view_7,
                                rhs_view_6,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_7 = T.tensor_view(
                                T.ptr_of(value_65[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_9.group, 4 @ scope_9.warp, 2, 8 @ scope_9.lane8, 32, 4 @ scope_9.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            lhs_view_8 = T.tensor_view(
                                T.ptr_of(value_70[0:0 + 128, 48:48 + 16]),
                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(128, 16),
                            )
                            rhs_view_7 = T.tensor_view(
                                T.ptr_of(value_76[48:48 + 16, 0:0 + 256]),
                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                acc_view_7,
                                lhs_view_8,
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
                        value_78 = T.tensor_view(
                            T.ptr_of(hidden), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
                        )
                        value_79 = T.tensor_view(
                            T.ptr_of(value_78), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_80 = T.tensor_view(
                            T.ptr_of(value_79), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_81 = T.tensor_view(
                            T.ptr_of(value_80[((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 1, bn_2 * 256:bn_2 * 256 + 1]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        value_77 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_82 = T.tensor_view(
                            T.ptr_of(value_77),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy_async_tensor(value_81, value_82)
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                    ) as scope_11:
                        value_84 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_77_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_85 = T.tensor_view(
                            T.ptr_of(value_77_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_86 = T.tensor_view(
                            T.ptr_of(value_85),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_87 = T.tensor_view(
                            T.ptr_of(value_86),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_88 = T.tensor_view(
                            T.ptr_of(value_84),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        dst_view_1 = T.tensor_view(
                            T.ptr_of(value_88[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.copy(value_87, dst_view_1)
                        value_89 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_90 = T.tensor_view(
                            T.ptr_of(acc_1),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_91 = T.tensor_view(
                            T.ptr_of(value_90),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_92 = T.tensor_view(
                            T.ptr_of(value_91),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        value_93 = T.tensor_view(
                            T.ptr_of(value_92),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        src_view_1 = T.tensor_view(
                            T.ptr_of(value_93[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(src_view_1, value_89, dtype="bf16")
                        y_1 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        lhs_view_9 = T.tensor_view(
                            T.ptr_of(value_88[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_11.group, 4 @ scope_11.warp, 2, 8 @ scope_11.lane8, 32, 4 @ scope_11.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.binary(lhs_view_9, value_89, y_1, kind=BinaryKind.ADD)
                        value_83 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_94 = T.tensor_view(
                            T.ptr_of(value_83),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(y_1, value_94)
                    with scope[:1, :32] as scope_12:
                        value_83_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_96 = T.tensor_view(
                            T.ptr_of(value_83_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_97 = T.tensor_view(
                            T.ptr_of(value_96),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_98 = T.tensor_view(
                            T.ptr_of(value_97),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_95_1 = T.tensor_view(
                            T.ptr_of(scratch_1[0:0 + 32768]),
                            layout=(128, 256),
                            shape=(128, 256),
                        )
                        value_99 = T.tensor_view(
                            T.ptr_of(value_95_1), layout=(128, 256), shape=(128, 256)
                        )
                        T.copy_async_tensor(value_98, value_99)
                        value_100 = T.tensor_view(
                            T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_101 = T.tensor_view(
                            T.ptr_of(value_100), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_102 = T.tensor_view(
                            T.ptr_of(value_101), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_103 = T.tensor_view(
                            T.ptr_of(value_102), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_104 = T.tensor_view(
                            T.ptr_of(value_103), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        value_105 = T.tensor_view(
                            T.ptr_of(value_104), layout=(1024, 5120), shape=(1024, 5120)
                        )
                        window_1 = T.tensor_view(
                            T.ptr_of(value_105[((0 * 8) + mi_2) * 128:((0 * 8) + mi_2) * 128 + 128, bn_2 * 256:bn_2 * 256 + 256]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        T.copy(value_99, window_1)
