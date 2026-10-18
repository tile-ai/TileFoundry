# analysis target=nvidia.h200_sxm module=QKV function=whole_b topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r144.81MB/w7.00MB@logical,r27.98GB/w924.00MB@total,r217.05MB/w7.00MB@cta,r213.52MB/w7.00MB@thread;rmem:r573.36MB/w565.85MB@logical,r93.70GB/w89.62GB@total,r726.90MB/w695.24MB@cta,r3.02MB/w2.74MB@thread;smem:r213.66MB/w144.91MB@logical,r28.65GB/w28.65GB@total,r222.25MB/w222.25MB@cta,r39.63MB/w210.05MB@thread footprint=cos:256.00KB;gamma_k:512B;gamma_q:512B;sin:256.00KB;v191:642:64.00KB;v193:643:8.25MB;w:288.00KB;x:256.00KB footprint-precision=exact peak=gmem:670.00MB;rmem:384.00KB;smem:208.00KB persistent=gmem:446.00MB
#   buffer=cos holds=51.31MB time=bn space=none reuse=13.75MB fits=no precision=exact
#   buffer=sin holds=51.31MB time=bn space=none reuse=13.75MB fits=no precision=exact
#   buffer=gamma_q holds=51.31MB time=bn space=cta.persistent reuse=3.61MB fits=no precision=exact
#   buffer=gamma_k holds=51.31MB time=bn space=cta.persistent reuse=3.61MB fits=no precision=exact
#   error="rmem placement peak 384.00KB exceeds capacity 256.00KB"
#   error="l2 reuse window bn holds 51.31MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole_b(
    out: Tensor[(8192, 14336), "bf16"], x: Tensor[(1, 8192, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(8192, 64), "bf16"], sin: Tensor[(8192, 64), "bf16"], out_1: Tensor[(8192, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(8192, 14336), shape=(8192, 14336))
            value_1 = T.tensor_view(T.ptr_of(out_1), layout=(8192, 14336), shape=(8192, 14336))
            T.copy(value, value_1)
            value_2_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_8_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for g in range(2, 3, 1):
                for bn in range(0, 56, 1):
                    for mi in range(((cta.persistent + (28 * g)) - (16 * bn)) - (132 * (((cta.persistent + (28 * g)) - (16 * bn)) // 132)), 16, 132):
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
                                value_3 = T.tensor_view(
                                    T.ptr_of(x), layout=(1, 8192, 5120), shape=(1, 8192, 5120)
                                )
                                value_4 = T.tensor_view(
                                    T.ptr_of(value_3), layout=(8192, 5120), shape=(8192, 5120)
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_4), layout=(8192, 5120), shape=(8192, 5120)
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(value_5[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                value_7 = T.tensor_view(
                                    T.ptr_of(value_2_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_6, value_7)
                                value_9 = T.tensor_view(
                                    T.ptr_of(w), layout=(14336, 5120), shape=(14336, 5120)
                                )
                                value_10 = T.tensor_view(
                                    T.ptr_of(value_9[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                value_11 = T.tensor_view(
                                    T.ptr_of(value_8_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                T.copy_async_tensor(value_10, value_11)
                            with Mesh(
                                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                            ) as scope_3:
                                value_12 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_13 = T.tensor_view(
                                    T.ptr_of(value_12),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_14 = T.tensor_view(
                                    T.ptr_of(value_13),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                acc_view = T.tensor_view(
                                    T.ptr_of(value_14[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_15 = T.tensor_view(
                                    T.ptr_of(value_2_stages[stage_counter % 3]),
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
                                value_19 = T.tensor_view(
                                    T.ptr_of(value_18),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_19[0:0 + 128, 0:0 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_20 = T.tensor_view(
                                    T.ptr_of(value_8_stages[stage_counter % 3]),
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
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                value_25 = T.tensor_view(
                                    T.ptr_of(value_24),
                                    layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_25[0:0 + 16, 0:0 + 256]),
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
                                    T.ptr_of(value_14[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_19[0:0 + 128, 16:16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_25[16:16 + 16, 0:0 + 256]),
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
                                    T.ptr_of(value_14[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_19[0:0 + 128, 32:32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_25[32:32 + 16, 0:0 + 256]),
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
                                    T.ptr_of(value_14[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_19[0:0 + 128, 48:48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_25[48:48 + 16, 0:0 + 256]),
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
                            value_27 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_28 = T.tensor_view(
                                T.ptr_of(acc),
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
                            value_31 = T.tensor_view(
                                T.ptr_of(value_30),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            src_view = T.tensor_view(
                                T.ptr_of(value_31[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(src_view, value_27, dtype="bf16")
                            value_26 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_32 = T.tensor_view(
                                T.ptr_of(value_26),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(value_27, value_32)
                        for _q in range(bn - (56 * (bn // 56)), 48, 1048576):
                            for _h in range(bn - (2 * (bn // 2)), 1, 2):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                                ) as scope_5:
                                    value_33 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_26_2 = T.tensor_view(
                                        147456,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_34 = T.tensor_view(
                                        T.ptr_of(value_26_2),
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
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_39 = T.tensor_view(
                                        T.ptr_of(value_38),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_40 = T.tensor_view(
                                        T.ptr_of(value_33),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view = T.tensor_view(
                                        T.ptr_of(value_40[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_39, dst_view)
                                    x32 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_1 = T.tensor_view(
                                        T.ptr_of(value_40[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_view_1, x32, dtype="f32")
                                    value_41 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_42 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_43 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, x32, value_43, kind=BinaryKind.MUL)
                                    value_44 = T.tensor_view(
                                        T.ptr_of(value_42),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_1 = T.tensor_view(
                                        T.ptr_of(value_44[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_43, dst_view_1)
                                    value_45 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_5.r, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    src_view_2 = T.tensor_view(
                                        T.ptr_of(value_44[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_5.r, 8, 16 @ scope_5.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.reduce(
                                        src_view_2,
                                        value_45,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_46 = T.tensor_view(
                                        T.ptr_of(value_41),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_view_2 = T.tensor_view(
                                        T.ptr_of(value_46[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_45, dst_view_2)
                                    value_47 = T.alloc_tensor(
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
                                        T.ptr_of(value_46[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_5.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(lhs_view_4, scalar, value_47, kind=BinaryKind.ADD)
                                    value_48 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_47, value_48, kind=UnaryKind.RSQRT)
                                    value_49 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, value_48, value_49, kind=BinaryKind.MUL)
                                    value_50 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_51 = T.tensor_view(
                                        T.ptr_of(gamma_q), layout=(256,), shape=(256,)
                                    )
                                    value_52 = T.tensor_view(
                                        T.ptr_of(value_51),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_53 = T.tensor_view(
                                        T.ptr_of(value_50),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_view_3 = T.tensor_view(
                                        T.ptr_of(value_53[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_52, dst_view_3)
                                    value_54 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_3 = T.tensor_view(
                                        T.ptr_of(value_53[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_5.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_view_3, value_54, dtype="f32")
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
                                    T.binary(value_54, scalar_1, gw, kind=BinaryKind.ADD)
                                    value_55 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_49, gw, value_55, kind=BinaryKind.MUL)
                                    nrm = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_55, nrm, dtype="bf16")
                                    value_56 = T.tensor_view(
                                        T.ptr_of(nrm),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 16),
                                        shape=(128, 256),
                                    )
                                    value_57 = T.tensor_view(
                                        T.ptr_of(value_37),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_58 = T.tensor_view(
                                        T.ptr_of(value_57),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_56, value_58)
                                    value_59 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_60 = T.tensor_view(
                                        T.ptr_of(value_58),
                                        layout=((128, 32), (64, 1)) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_61 = T.tensor_view(
                                        T.ptr_of(value_60),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_62 = T.tensor_view(
                                        T.ptr_of(value_59),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_4 = T.tensor_view(
                                        T.ptr_of(value_62[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_61, dst_view_4)
                                    lo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_4 = T.tensor_view(
                                        T.ptr_of(value_62[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_4, lo, dtype="f32")
                                    value_63 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_64 = T.tensor_view(
                                        T.ptr_of(cos), layout=(8192, 64), shape=(8192, 64)
                                    )
                                    value_65 = T.tensor_view(
                                        T.ptr_of(value_64[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_66 = T.tensor_view(
                                        T.ptr_of(value_65),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_67 = T.tensor_view(
                                        T.ptr_of(value_63),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_5 = T.tensor_view(
                                        T.ptr_of(value_67[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_66, dst_view_5)
                                    cl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_5 = T.tensor_view(
                                        T.ptr_of(value_67[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_5, cl, dtype="f32")
                                    value_68 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, cl, value_68, kind=BinaryKind.MUL)
                                    value_69 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_68, value_69, dtype="bf16")
                                    value_70 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_69, value_70, dtype="f32")
                                    value_71 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_72 = T.tensor_view(
                                        T.ptr_of(value_58),
                                        layout=((128, 32), (64, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_73 = T.tensor_view(
                                        T.ptr_of(value_72),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_74 = T.tensor_view(
                                        T.ptr_of(value_71),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_6 = T.tensor_view(
                                        T.ptr_of(value_74[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_73, dst_view_6)
                                    hi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_6 = T.tensor_view(
                                        T.ptr_of(value_74[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_6, hi, dtype="f32")
                                    value_75 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi, value_75, kind=UnaryKind.NEG)
                                    value_76 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_77 = T.tensor_view(
                                        T.ptr_of(sin), layout=(8192, 64), shape=(8192, 64)
                                    )
                                    value_78 = T.tensor_view(
                                        T.ptr_of(value_77[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_79 = T.tensor_view(
                                        T.ptr_of(value_78),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_80 = T.tensor_view(
                                        T.ptr_of(value_76),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_7 = T.tensor_view(
                                        T.ptr_of(value_80[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_79, dst_view_7)
                                    sl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_7 = T.tensor_view(
                                        T.ptr_of(value_80[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_7, sl, dtype="f32")
                                    value_81 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_75, sl, value_81, kind=BinaryKind.MUL)
                                    value_82 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_81, value_82, dtype="bf16")
                                    value_83 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_82, value_83, dtype="f32")
                                    value_84 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_70, value_83, value_84, kind=BinaryKind.ADD)
                                    rlo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_84, rlo, dtype="bf16")
                                    value_85 = T.tensor_view(
                                        T.ptr_of(rlo),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_86 = T.tensor_view(
                                        T.ptr_of(value_58),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window = T.tensor_view(
                                        T.ptr_of(value_86[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_85, window)
                                    value_87 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_88 = T.tensor_view(
                                        T.ptr_of(value_64[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_89 = T.tensor_view(
                                        T.ptr_of(value_88),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_90 = T.tensor_view(
                                        T.ptr_of(value_87),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_8 = T.tensor_view(
                                        T.ptr_of(value_90[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_89, dst_view_8)
                                    ch = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_8 = T.tensor_view(
                                        T.ptr_of(value_90[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_8, ch, dtype="f32")
                                    value_91 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi, ch, value_91, kind=BinaryKind.MUL)
                                    value_92 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_91, value_92, dtype="bf16")
                                    value_93 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_92, value_93, dtype="f32")
                                    value_94 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_95 = T.tensor_view(
                                        T.ptr_of(value_77[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_96 = T.tensor_view(
                                        T.ptr_of(value_95),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_97 = T.tensor_view(
                                        T.ptr_of(value_94),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_9 = T.tensor_view(
                                        T.ptr_of(value_97[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_96, dst_view_9)
                                    sh = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_9 = T.tensor_view(
                                        T.ptr_of(value_97[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_9, sh, dtype="f32")
                                    value_98 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, sh, value_98, kind=BinaryKind.MUL)
                                    value_99 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_98, value_99, dtype="bf16")
                                    value_100 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_99, value_100, dtype="f32")
                                    value_101 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_93,
                                        value_100,
                                        value_101,
                                        kind=BinaryKind.ADD,
                                    )
                                    rhi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_101, rhi, dtype="bf16")
                                    value_102 = T.tensor_view(
                                        T.ptr_of(rhi),
                                        layout=(16 @ scope_5.r, 8, 16 @ scope_5.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_103 = T.tensor_view(
                                        T.ptr_of(value_86),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_1 = T.tensor_view(
                                        T.ptr_of(value_103[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_102, window_1)
                        for _k0 in range(47, bn - (56 * (bn // 56)), 1048576):
                            for _k1 in range(bn - (56 * (bn // 56)), 52, 1048576):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("r", "c")
                                ) as scope_6:
                                    value_104 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_26_3 = T.tensor_view(
                                        147456,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_105 = T.tensor_view(
                                        T.ptr_of(value_26_3),
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
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_112 = T.tensor_view(
                                        T.ptr_of(value_111),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_113 = T.tensor_view(
                                        T.ptr_of(value_104),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_10 = T.tensor_view(
                                        T.ptr_of(value_113[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_112, dst_view_10)
                                    x32_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_10 = T.tensor_view(
                                        T.ptr_of(value_113[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_view_10, x32_1, dtype="f32")
                                    value_114 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_115 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_116 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, x32_1, value_116, kind=BinaryKind.MUL)
                                    value_117 = T.tensor_view(
                                        T.ptr_of(value_115),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_view_11 = T.tensor_view(
                                        T.ptr_of(value_117[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_116, dst_view_11)
                                    value_118 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_6.r, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    src_view_11 = T.tensor_view(
                                        T.ptr_of(value_117[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ scope_6.r, 8, 16 @ scope_6.c, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.reduce(
                                        src_view_11,
                                        value_118,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_119 = T.tensor_view(
                                        T.ptr_of(value_114),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_view_12 = T.tensor_view(
                                        T.ptr_of(value_119[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_118, dst_view_12)
                                    value_120 = T.alloc_tensor(
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
                                        T.ptr_of(value_119[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ scope_6.r, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(
                                        lhs_view_5,
                                        scalar_2,
                                        value_120,
                                        kind=BinaryKind.ADD,
                                    )
                                    value_121 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.r, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_120, value_121, kind=UnaryKind.RSQRT)
                                    value_122 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, value_121, value_122, kind=BinaryKind.MUL)
                                    value_123 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_124 = T.tensor_view(
                                        T.ptr_of(gamma_k), layout=(256,), shape=(256,)
                                    )
                                    value_125 = T.tensor_view(
                                        T.ptr_of(value_124),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_126 = T.tensor_view(
                                        T.ptr_of(value_123),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_view_13 = T.tensor_view(
                                        T.ptr_of(value_126[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_125, dst_view_13)
                                    value_127 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    src_view_12 = T.tensor_view(
                                        T.ptr_of(value_126[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ scope_6.c, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_view_12, value_127, dtype="f32")
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
                                    T.binary(value_127, scalar_3, gw_1, kind=BinaryKind.ADD)
                                    value_128 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_122, gw_1, value_128, kind=BinaryKind.MUL)
                                    nrm_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_128, nrm_1, dtype="bf16")
                                    value_129 = T.tensor_view(
                                        T.ptr_of(nrm_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 16),
                                        shape=(128, 256),
                                    )
                                    value_130 = T.tensor_view(
                                        T.ptr_of(value_110),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_131 = T.tensor_view(
                                        T.ptr_of(value_130),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_129, value_131)
                                    value_132 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_133 = T.tensor_view(
                                        T.ptr_of(value_131),
                                        layout=((128, 32), (64, 1)) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_134 = T.tensor_view(
                                        T.ptr_of(value_133),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_135 = T.tensor_view(
                                        T.ptr_of(value_132),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_14 = T.tensor_view(
                                        T.ptr_of(value_135[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_134, dst_view_14)
                                    lo_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_13 = T.tensor_view(
                                        T.ptr_of(value_135[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_13, lo_1, dtype="f32")
                                    value_136 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_137 = T.tensor_view(
                                        T.ptr_of(cos), layout=(8192, 64), shape=(8192, 64)
                                    )
                                    value_138 = T.tensor_view(
                                        T.ptr_of(value_137[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_139 = T.tensor_view(
                                        T.ptr_of(value_138),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_140 = T.tensor_view(
                                        T.ptr_of(value_136),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_15 = T.tensor_view(
                                        T.ptr_of(value_140[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_139, dst_view_15)
                                    cl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_14 = T.tensor_view(
                                        T.ptr_of(value_140[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_14, cl_1, dtype="f32")
                                    value_141 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, cl_1, value_141, kind=BinaryKind.MUL)
                                    value_142 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_141, value_142, dtype="bf16")
                                    value_143 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_142, value_143, dtype="f32")
                                    value_144 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_145 = T.tensor_view(
                                        T.ptr_of(value_131),
                                        layout=((128, 32), (64, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    value_146 = T.tensor_view(
                                        T.ptr_of(value_145),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_147 = T.tensor_view(
                                        T.ptr_of(value_144),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_16 = T.tensor_view(
                                        T.ptr_of(value_147[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_146, dst_view_16)
                                    hi_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_15 = T.tensor_view(
                                        T.ptr_of(value_147[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_15, hi_1, dtype="f32")
                                    value_148 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi_1, value_148, kind=UnaryKind.NEG)
                                    value_149 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_150 = T.tensor_view(
                                        T.ptr_of(sin), layout=(8192, 64), shape=(8192, 64)
                                    )
                                    value_151 = T.tensor_view(
                                        T.ptr_of(value_150[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_152 = T.tensor_view(
                                        T.ptr_of(value_151),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_153 = T.tensor_view(
                                        T.ptr_of(value_149),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_17 = T.tensor_view(
                                        T.ptr_of(value_153[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_152, dst_view_17)
                                    sl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_16 = T.tensor_view(
                                        T.ptr_of(value_153[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_16, sl_1, dtype="f32")
                                    value_154 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_148, sl_1, value_154, kind=BinaryKind.MUL)
                                    value_155 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_154, value_155, dtype="bf16")
                                    value_156 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_155, value_156, dtype="f32")
                                    value_157 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_143,
                                        value_156,
                                        value_157,
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
                                    T.cast(value_157, rlo_1, dtype="bf16")
                                    value_158 = T.tensor_view(
                                        T.ptr_of(rlo_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_159 = T.tensor_view(
                                        T.ptr_of(value_131),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_2 = T.tensor_view(
                                        T.ptr_of(value_159[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_158, window_2)
                                    value_160 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_161 = T.tensor_view(
                                        T.ptr_of(value_137[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_162 = T.tensor_view(
                                        T.ptr_of(value_161),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_163 = T.tensor_view(
                                        T.ptr_of(value_160),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_18 = T.tensor_view(
                                        T.ptr_of(value_163[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_162, dst_view_18)
                                    ch_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_17 = T.tensor_view(
                                        T.ptr_of(value_163[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_17, ch_1, dtype="f32")
                                    value_164 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi_1, ch_1, value_164, kind=BinaryKind.MUL)
                                    value_165 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_164, value_165, dtype="bf16")
                                    value_166 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_165, value_166, dtype="f32")
                                    value_167 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_168 = T.tensor_view(
                                        T.ptr_of(value_150[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((128, 32), (64, 1)),
                                        shape=(128, 32),
                                    )
                                    value_169 = T.tensor_view(
                                        T.ptr_of(value_168),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_170 = T.tensor_view(
                                        T.ptr_of(value_167),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_view_19 = T.tensor_view(
                                        T.ptr_of(value_170[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_169, dst_view_19)
                                    sh_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    src_view_18 = T.tensor_view(
                                        T.ptr_of(value_170[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_view_18, sh_1, dtype="f32")
                                    value_171 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, sh_1, value_171, kind=BinaryKind.MUL)
                                    value_172 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_171, value_172, dtype="bf16")
                                    value_173 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_172, value_173, dtype="f32")
                                    value_174 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_166,
                                        value_173,
                                        value_174,
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
                                    T.cast(value_174, rhi_1, dtype="bf16")
                                    value_175 = T.tensor_view(
                                        T.ptr_of(rhi_1),
                                        layout=(16 @ scope_6.r, 8, 16 @ scope_6.c, 2),
                                        shape=(128, 32),
                                    )
                                    value_176 = T.tensor_view(
                                        T.ptr_of(value_159),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_3 = T.tensor_view(
                                        T.ptr_of(value_176[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_175, window_3)
                        with scope[:1, :32] as scope_7:
                            value_26_4 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_177 = T.tensor_view(
                                T.ptr_of(value_26_4),
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
                            value_184 = T.tensor_view(
                                T.ptr_of(out_1), layout=(8192, 14336), shape=(8192, 14336)
                            )
                            value_185 = T.tensor_view(
                                T.ptr_of(value_184), layout=(8192, 14336), shape=(8192, 14336)
                            )
                            value_186 = T.tensor_view(
                                T.ptr_of(value_185), layout=(8192, 14336), shape=(8192, 14336)
                            )
                            value_187 = T.tensor_view(
                                T.ptr_of(value_186), layout=(8192, 14336), shape=(8192, 14336)
                            )
                            value_188 = T.tensor_view(
                                T.ptr_of(value_187[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 1, bn * 256:bn * 256 + 1]),
                                layout=((128, 256), (14336, 1)),
                                shape=(128, 256),
                            )
                            value_189 = T.tensor_view(
                                T.ptr_of(value_188),
                                layout=((128, 256), (14336, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(value_183, value_189)
