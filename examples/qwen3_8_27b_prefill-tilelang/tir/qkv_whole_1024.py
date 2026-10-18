# analysis target=nvidia.h200_sxm module=QKV function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r144.81MB/w35.00MB@logical,r27.98GB/w4.51GB@total,r217.05MB/w35.00MB@cta,r213.52MB/w35.00MB@thread;rmem:r573.36MB/w565.85MB@logical,r93.70GB/w89.62GB@total,r726.90MB/w695.24MB@cta,r3.02MB/w2.74MB@thread;smem:r213.66MB/w144.91MB@logical,r28.65GB/w28.65GB@total,r222.25MB/w222.25MB@cta,r39.63MB/w210.05MB@thread footprint=cos:0;gamma_k:0;gamma_q:0;sin:0;v192:570:0;v194:571:0;w:0;x:0 footprint-precision=lower_bound peak=gmem:178.31MB;rmem:384.00KB;smem:208.00KB persistent=gmem:150.25MB
#   error="rmem placement peak 384.00KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(1024, 64), "bf16"], sin: Tensor[(1024, 64), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(1024, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(1024, 14336), shape=(1024, 14336))
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
                for bn in range(0, 56, 1):
                    for mi in range(((cta.persistent - (52 * g)) - (8 * bn)) - (132 * (((cta.persistent - (52 * g)) - (8 * bn)) // 132)), 8, 132):
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
                                    T.ptr_of(w), layout=(14336, 5120), shape=(14336, 5120)
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
                        for _q in range(bn - (56 * (bn // 56)), 48, 1048576):
                            for _h in range(bn - (2 * (bn // 2)), 1, 2):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                                ) as scope_5:
                                    value_32 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
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
                                        T.ptr_of(value_35),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_37 = T.tensor_view(
                                        T.ptr_of(value_36),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_38 = T.tensor_view(
                                        T.ptr_of(value_37),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_39 = T.tensor_view(
                                        T.ptr_of(value_32),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view = T.tensor_view(
                                        T.ptr_of(value_39[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_38, dst_view)
                                    x32 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_1 = T.tensor_view(
                                        T.ptr_of(value_39[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_view_1, x32, dtype="f32")
                                    value_40 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_41 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_42 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, x32, value_42, kind=BinaryKind.MUL)
                                    value_43 = T.tensor_view(
                                        T.ptr_of(value_41),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_1 = T.tensor_view(
                                        T.ptr_of(value_43[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_42, dst_view_1)
                                    value_44 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_5.r, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    src_view_2 = T.tensor_view(
                                        T.ptr_of(value_43[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.reduce(
                                        src_view_2,
                                        value_44,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_45 = T.tensor_view(
                                        T.ptr_of(value_40),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_view_2 = T.tensor_view(
                                        T.ptr_of(value_45[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_44, dst_view_2)
                                    value_46 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    scalar = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar, 1e-06)
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_45[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(lhs_view_4, scalar, value_46, kind=BinaryKind.ADD)
                                    value_47 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_46, value_47, kind=UnaryKind.RSQRT)
                                    value_48 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, value_47, value_48, kind=BinaryKind.MUL)
                                    value_49 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_50 = T.tensor_view(
                                        T.ptr_of(gamma_q), layout=(256,), shape=(256,)
                                    )
                                    value_51 = T.tensor_view(
                                        T.ptr_of(value_50),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_52 = T.tensor_view(
                                        T.ptr_of(value_49),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_view_3 = T.tensor_view(
                                        T.ptr_of(value_52[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_51, dst_view_3)
                                    value_53 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_3 = T.tensor_view(
                                        T.ptr_of(value_52[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_view_3, value_53, dtype="f32")
                                    gw = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    scalar_1 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_1, 1.0)
                                    T.binary(value_53, scalar_1, gw, kind=BinaryKind.ADD)
                                    value_54 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_48, gw, value_54, kind=BinaryKind.MUL)
                                    nrm = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_54, nrm, dtype="bf16")
                                    value_55 = T.tensor_view(
                                        T.ptr_of(nrm),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(128, 256),
                                    )
                                    value_56 = T.tensor_view(
                                        T.ptr_of(value_36),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_57 = T.tensor_view(
                                        T.ptr_of(value_56),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_55, value_57)
                                    value_58 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_59 = T.tensor_view(
                                        T.ptr_of(value_57),
                                        layout=((128, 32), (64, 1)) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_60 = T.tensor_view(
                                        T.ptr_of(value_59),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_61 = T.tensor_view(
                                        T.ptr_of(value_58),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_4 = T.tensor_view(
                                        T.ptr_of(value_61[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_60, dst_view_4)
                                    lo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_4 = T.tensor_view(
                                        T.ptr_of(value_61[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_4, lo, dtype="f32")
                                    value_62 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_63 = T.tensor_view(
                                        T.ptr_of(cos), layout=(1024, 64), shape=(1024, 64)
                                    )
                                    value_64 = T.tensor_view(
                                        T.ptr_of(value_63[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_65 = T.tensor_view(
                                        T.ptr_of(value_64),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_66 = T.tensor_view(
                                        T.ptr_of(value_62),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_5 = T.tensor_view(
                                        T.ptr_of(value_66[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_65, dst_view_5)
                                    cl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_5 = T.tensor_view(
                                        T.ptr_of(value_66[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_5, cl, dtype="f32")
                                    value_67 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, cl, value_67, kind=BinaryKind.MUL)
                                    value_68 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_67, value_68, dtype="bf16")
                                    value_69 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_68, value_69, dtype="f32")
                                    value_70 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_71 = T.tensor_view(
                                        T.ptr_of(value_57),
                                        layout=((128, 32), (64, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_72 = T.tensor_view(
                                        T.ptr_of(value_71),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_73 = T.tensor_view(
                                        T.ptr_of(value_70),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_6 = T.tensor_view(
                                        T.ptr_of(value_73[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_72, dst_view_6)
                                    hi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_6 = T.tensor_view(
                                        T.ptr_of(value_73[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_6, hi, dtype="f32")
                                    value_74 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi, value_74, kind=UnaryKind.NEG)
                                    value_75 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_76 = T.tensor_view(
                                        T.ptr_of(sin), layout=(1024, 64), shape=(1024, 64)
                                    )
                                    value_77 = T.tensor_view(
                                        T.ptr_of(value_76[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_78 = T.tensor_view(
                                        T.ptr_of(value_77),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_79 = T.tensor_view(
                                        T.ptr_of(value_75),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_7 = T.tensor_view(
                                        T.ptr_of(value_79[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_78, dst_view_7)
                                    sl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_7 = T.tensor_view(
                                        T.ptr_of(value_79[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_7, sl, dtype="f32")
                                    value_80 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_74, sl, value_80, kind=BinaryKind.MUL)
                                    value_81 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_80, value_81, dtype="bf16")
                                    value_82 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_81, value_82, dtype="f32")
                                    value_83 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_69, value_82, value_83, kind=BinaryKind.ADD)
                                    rlo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_83, rlo, dtype="bf16")
                                    value_84 = T.tensor_view(
                                        T.ptr_of(rlo),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_85 = T.tensor_view(
                                        T.ptr_of(value_57),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window = T.tensor_view(
                                        T.ptr_of(value_85[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_84, window)
                                    value_86 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_87 = T.tensor_view(
                                        T.ptr_of(value_63[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_88 = T.tensor_view(
                                        T.ptr_of(value_87),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_89 = T.tensor_view(
                                        T.ptr_of(value_86),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_8 = T.tensor_view(
                                        T.ptr_of(value_89[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_88, dst_view_8)
                                    ch = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_8 = T.tensor_view(
                                        T.ptr_of(value_89[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_8, ch, dtype="f32")
                                    value_90 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi, ch, value_90, kind=BinaryKind.MUL)
                                    value_91 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_90, value_91, dtype="bf16")
                                    value_92 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_91, value_92, dtype="f32")
                                    value_93 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_94 = T.tensor_view(
                                        T.ptr_of(value_76[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_95 = T.tensor_view(
                                        T.ptr_of(value_94),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_96 = T.tensor_view(
                                        T.ptr_of(value_93),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_9 = T.tensor_view(
                                        T.ptr_of(value_96[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_95, dst_view_9)
                                    sh = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_9 = T.tensor_view(
                                        T.ptr_of(value_96[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_9, sh, dtype="f32")
                                    value_97 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, sh, value_97, kind=BinaryKind.MUL)
                                    value_98 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_97, value_98, dtype="bf16")
                                    value_99 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_98, value_99, dtype="f32")
                                    value_100 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_92, value_99, value_100, kind=BinaryKind.ADD)
                                    rhi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_100, rhi, dtype="bf16")
                                    value_101 = T.tensor_view(
                                        T.ptr_of(rhi),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_102 = T.tensor_view(
                                        T.ptr_of(value_85),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_1 = T.tensor_view(
                                        T.ptr_of(value_102[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_101, window_1)
                        for _k0 in range(47, bn - (56 * (bn // 56)), 1048576):
                            for _k1 in range(bn - (56 * (bn // 56)), 52, 1048576):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                                ) as scope_6:
                                    value_103 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_25_3 = T.tensor_view(
                                        147456,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_104 = T.tensor_view(
                                        T.ptr_of(value_25_3),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_105 = T.tensor_view(
                                        T.ptr_of(value_104),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_106 = T.tensor_view(
                                        T.ptr_of(value_105),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_107 = T.tensor_view(
                                        T.ptr_of(value_106),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_108 = T.tensor_view(
                                        T.ptr_of(value_107),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_109 = T.tensor_view(
                                        T.ptr_of(value_108),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_110 = T.tensor_view(
                                        T.ptr_of(value_109),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_111 = T.tensor_view(
                                        T.ptr_of(value_110),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_112 = T.tensor_view(
                                        T.ptr_of(value_103),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_10 = T.tensor_view(
                                        T.ptr_of(value_112[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_111, dst_view_10)
                                    x32_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_10 = T.tensor_view(
                                        T.ptr_of(value_112[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_view_10, x32_1, dtype="f32")
                                    value_113 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_114 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_115 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, x32_1, value_115, kind=BinaryKind.MUL)
                                    value_116 = T.tensor_view(
                                        T.ptr_of(value_114),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_11 = T.tensor_view(
                                        T.ptr_of(value_116[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_115, dst_view_11)
                                    value_117 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_6.r, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    src_view_11 = T.tensor_view(
                                        T.ptr_of(value_116[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.reduce(
                                        src_view_11,
                                        value_117,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_118 = T.tensor_view(
                                        T.ptr_of(value_113),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_view_12 = T.tensor_view(
                                        T.ptr_of(value_118[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_117, dst_view_12)
                                    value_119 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    scalar_2 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_2, 1e-06)
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(value_118[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(
                                        lhs_view_5,
                                        scalar_2,
                                        value_119,
                                        kind=BinaryKind.ADD,
                                    )
                                    value_120 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_119, value_120, kind=UnaryKind.RSQRT)
                                    value_121 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, value_120, value_121, kind=BinaryKind.MUL)
                                    value_122 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_123 = T.tensor_view(
                                        T.ptr_of(gamma_k), layout=(256,), shape=(256,)
                                    )
                                    value_124 = T.tensor_view(
                                        T.ptr_of(value_123),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_125 = T.tensor_view(
                                        T.ptr_of(value_122),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_view_13 = T.tensor_view(
                                        T.ptr_of(value_125[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_124, dst_view_13)
                                    value_126 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_12 = T.tensor_view(
                                        T.ptr_of(value_125[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_view_12, value_126, dtype="f32")
                                    gw_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    scalar_3 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_3, 1.0)
                                    T.binary(value_126, scalar_3, gw_1, kind=BinaryKind.ADD)
                                    value_127 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_121, gw_1, value_127, kind=BinaryKind.MUL)
                                    nrm_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_127, nrm_1, dtype="bf16")
                                    value_128 = T.tensor_view(
                                        T.ptr_of(nrm_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(128, 256),
                                    )
                                    value_129 = T.tensor_view(
                                        T.ptr_of(value_109),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_130 = T.tensor_view(
                                        T.ptr_of(value_129),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_128, value_130)
                                    value_131 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_132 = T.tensor_view(
                                        T.ptr_of(value_130),
                                        layout=((128, 32), (64, 1)) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_133 = T.tensor_view(
                                        T.ptr_of(value_132),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_134 = T.tensor_view(
                                        T.ptr_of(value_131),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_14 = T.tensor_view(
                                        T.ptr_of(value_134[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_133, dst_view_14)
                                    lo_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_13 = T.tensor_view(
                                        T.ptr_of(value_134[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_13, lo_1, dtype="f32")
                                    value_135 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_136 = T.tensor_view(
                                        T.ptr_of(cos), layout=(1024, 64), shape=(1024, 64)
                                    )
                                    value_137 = T.tensor_view(
                                        T.ptr_of(value_136[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_138 = T.tensor_view(
                                        T.ptr_of(value_137),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_139 = T.tensor_view(
                                        T.ptr_of(value_135),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_15 = T.tensor_view(
                                        T.ptr_of(value_139[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_138, dst_view_15)
                                    cl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_14 = T.tensor_view(
                                        T.ptr_of(value_139[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_14, cl_1, dtype="f32")
                                    value_140 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, cl_1, value_140, kind=BinaryKind.MUL)
                                    value_141 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_140, value_141, dtype="bf16")
                                    value_142 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_141, value_142, dtype="f32")
                                    value_143 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_144 = T.tensor_view(
                                        T.ptr_of(value_130),
                                        layout=((128, 32), (64, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_145 = T.tensor_view(
                                        T.ptr_of(value_144),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_146 = T.tensor_view(
                                        T.ptr_of(value_143),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_16 = T.tensor_view(
                                        T.ptr_of(value_146[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_145, dst_view_16)
                                    hi_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_15 = T.tensor_view(
                                        T.ptr_of(value_146[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_15, hi_1, dtype="f32")
                                    value_147 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi_1, value_147, kind=UnaryKind.NEG)
                                    value_148 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_149 = T.tensor_view(
                                        T.ptr_of(sin), layout=(1024, 64), shape=(1024, 64)
                                    )
                                    value_150 = T.tensor_view(
                                        T.ptr_of(value_149[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_151 = T.tensor_view(
                                        T.ptr_of(value_150),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_152 = T.tensor_view(
                                        T.ptr_of(value_148),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_17 = T.tensor_view(
                                        T.ptr_of(value_152[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_151, dst_view_17)
                                    sl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_16 = T.tensor_view(
                                        T.ptr_of(value_152[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_16, sl_1, dtype="f32")
                                    value_153 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_147, sl_1, value_153, kind=BinaryKind.MUL)
                                    value_154 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_153, value_154, dtype="bf16")
                                    value_155 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_154, value_155, dtype="f32")
                                    value_156 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_142,
                                        value_155,
                                        value_156,
                                        kind=BinaryKind.ADD,
                                    )
                                    rlo_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_156, rlo_1, dtype="bf16")
                                    value_157 = T.tensor_view(
                                        T.ptr_of(rlo_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_158 = T.tensor_view(
                                        T.ptr_of(value_130),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_2 = T.tensor_view(
                                        T.ptr_of(value_158[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_157, window_2)
                                    value_159 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_160 = T.tensor_view(
                                        T.ptr_of(value_136[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_161 = T.tensor_view(
                                        T.ptr_of(value_160),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_162 = T.tensor_view(
                                        T.ptr_of(value_159),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_18 = T.tensor_view(
                                        T.ptr_of(value_162[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_161, dst_view_18)
                                    ch_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_17 = T.tensor_view(
                                        T.ptr_of(value_162[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_17, ch_1, dtype="f32")
                                    value_163 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi_1, ch_1, value_163, kind=BinaryKind.MUL)
                                    value_164 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_163, value_164, dtype="bf16")
                                    value_165 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_164, value_165, dtype="f32")
                                    value_166 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_167 = T.tensor_view(
                                        T.ptr_of(value_149[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_168 = T.tensor_view(
                                        T.ptr_of(value_167),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_169 = T.tensor_view(
                                        T.ptr_of(value_166),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_19 = T.tensor_view(
                                        T.ptr_of(value_169[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_168, dst_view_19)
                                    sh_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_18 = T.tensor_view(
                                        T.ptr_of(value_169[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_18, sh_1, dtype="f32")
                                    value_170 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, sh_1, value_170, kind=BinaryKind.MUL)
                                    value_171 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_170, value_171, dtype="bf16")
                                    value_172 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_171, value_172, dtype="f32")
                                    value_173 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_165,
                                        value_172,
                                        value_173,
                                        kind=BinaryKind.ADD,
                                    )
                                    rhi_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_173, rhi_1, dtype="bf16")
                                    value_174 = T.tensor_view(
                                        T.ptr_of(rhi_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_175 = T.tensor_view(
                                        T.ptr_of(value_158),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_3 = T.tensor_view(
                                        T.ptr_of(value_175[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_174, window_3)
                        with scope[:1, :32] as scope_7:
                            value_25_4 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_177 = T.tensor_view(
                                T.ptr_of(value_25_4),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_178 = T.tensor_view(
                                T.ptr_of(value_177),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_179 = T.tensor_view(
                                T.ptr_of(value_178),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_180 = T.tensor_view(
                                T.ptr_of(value_179),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_181 = T.tensor_view(
                                T.ptr_of(value_180),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_182 = T.tensor_view(
                                T.ptr_of(value_181),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_183 = T.tensor_view(
                                T.ptr_of(value_182),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_176_1 = T.tensor_view(
                                T.ptr_of(scratch[0:0 + 32768]),
                                layout=(128, 256),
                                shape=(128, 256),
                            )
                            value_184 = T.tensor_view(
                                T.ptr_of(value_176_1), layout=(128, 256), shape=(128, 256)
                            )
                            T.copy_async_tensor(value_183, value_184)
                            value_185 = T.tensor_view(
                                T.ptr_of(out), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            value_186 = T.tensor_view(
                                T.ptr_of(value_185), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            value_187 = T.tensor_view(
                                T.ptr_of(value_186), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            value_188 = T.tensor_view(
                                T.ptr_of(value_187), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            value_189 = T.tensor_view(
                                T.ptr_of(value_188), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            window_4 = T.tensor_view(
                                T.ptr_of(value_189[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                                layout=((128, 256), (14336, 1)),
                                shape=(128, 256),
                            )
                            T.copy(value_184, window_4)
