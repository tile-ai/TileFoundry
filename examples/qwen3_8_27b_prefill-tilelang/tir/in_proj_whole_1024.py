# analysis target=nvidia.h200_sxm module=InProj function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r167.81MB/w40.62MB@logical,r31.94GB/w5.24GB@total,r247.81MB/w40.62MB@cta,r247.81MB/w40.62MB@thread;rmem:r662.27MB/w654.19MB@logical,r85.38GB/w85.36GB@total,r662.35MB/w662.19MB@cta,r2.75MB/w2.59MB@thread;smem:r247.81MB/w167.81MB@logical,r31.94GB/w31.94GB@total,r247.81MB/w247.81MB@cta,r45.96MB/w243.77MB@thread footprint=v32:426:64.00KB;v34:427:8.25MB;w:544.00KB;x:128.00KB footprint-precision=exact peak=gmem:205.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:172.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 5120), "bf16"], w: Tensor[(16640, 5120), "bf16"], out: Tensor[(1024, 16640), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(1024, 16640), shape=(1024, 16640))
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
            for g in range(0, 1, 1):
                for bn in range(0, 65, 1):
                    for mi in range(((cta.persistent + (8 * g)) - (8 * bn)) - (132 * (((cta.persistent + (8 * g)) - (8 * bn)) // 132)), 8, 132):
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
                                    T.ptr_of(x), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
                                )
                                value_3 = T.tensor_view(
                                    T.ptr_of(value_2), layout=(1024, 5120), shape=(1024, 5120)
                                )
                                value_4 = T.tensor_view(
                                    T.ptr_of(value_3), layout=(1024, 5120), shape=(1024, 5120)
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_4[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, k:k + 1]),
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
                                    T.ptr_of(w), layout=(16640, 5120), shape=(16640, 5120)
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
                        with scope[:1, :32] as scope_5:
                            value_25_2 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_32 = T.tensor_view(
                                T.ptr_of(value_25_2),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_33 = T.tensor_view(
                                T.ptr_of(value_32),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_34 = T.tensor_view(
                                T.ptr_of(value_33),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_35 = T.tensor_view(
                                T.ptr_of(out), layout=(1024, 16640), shape=(1024, 16640)
                            )
                            value_36 = T.tensor_view(
                                T.ptr_of(value_35), layout=(1024, 16640), shape=(1024, 16640)
                            )
                            value_37 = T.tensor_view(
                                T.ptr_of(value_36), layout=(1024, 16640), shape=(1024, 16640)
                            )
                            value_38 = T.tensor_view(
                                T.ptr_of(value_37), layout=(1024, 16640), shape=(1024, 16640)
                            )
                            value_39 = T.tensor_view(
                                T.ptr_of(value_38), layout=(1024, 16640), shape=(1024, 16640)
                            )
                            value_40 = T.tensor_view(
                                T.ptr_of(value_39[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, bn * 256:bn * 256 + 1]),
                                layout=((128, 256), (16640, 1)),
                                shape=(128, 256),
                            )
                            value_41 = T.tensor_view(
                                T.ptr_of(value_40),
                                layout=((128, 256), (16640, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(value_34, value_41)
