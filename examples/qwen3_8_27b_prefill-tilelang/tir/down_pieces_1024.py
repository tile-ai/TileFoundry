# analysis target=nvidia.h200_sxm module=Residual function=pieces topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r17.00MB/w20.00MB@logical,r13.15GB/w2.58GB@total,r102.00MB/w20.00MB@cta,r102.00MB/w16.02MB@thread;rmem:r276.02MB/w272.12MB@logical,r35.60GB/w35.58GB@total,r276.17MB/w276.00MB@cta,r1.25MB/w1.08MB@thread;smem:r102.00MB/w17.00MB@logical,r13.15GB/w13.15GB@total,r102.00MB/w102.00MB@cta,r17.53MB/w102.00MB@thread footprint=v40:124:16.00MB;w:512.00KB;x:512.00KB footprint-precision=exact peak=gmem:220.00MB;rmem:128.00KB;smem:144.00KB persistent=gmem:204.00MB
#   buffer=w holds=84.00MB time=mi space=none reuse=238.00MB fits=no precision=exact
#   buffer=x holds=84.00MB time=bb space=none reuse=102.00MB fits=no precision=exact
#   error="l2 reuse window mi holds 84.00MB at a 132-unit wave, exceeding capacity 47.68MB"
#   error="l2 reuse window bb holds 84.00MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pieces(
    x: Tensor[(1, 1024, 17408), "bf16"], w: ConstTensor[(5120, 17408), "bf16"], out: Tensor[(16384, 256), "f32"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("persistent",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(16384, 256), shape=(16384, 256))
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
            for bb in range(16, 20, 1):
                for mi in range(0, 8, 1):
                    for pc in range((((cta.persistent - (32 * bb)) - (4 * mi)) - (132 * ((((cta.persistent - (32 * bb)) - (4 * mi)) - 16) // 132))) - 16, 4, 132):
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
                        for kk in range(0, 68, 1):
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
                                    T.ptr_of(value_4[((0 * 8) + mi) * 128:((0 * 8) + mi) * 128 + 1, ((pc * 68) + kk) * 64:((pc * 68) + kk) * 64 + 1]),
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
                                    T.ptr_of(value_8[bb * 256:bb * 256 + 1, ((pc * 68) + kk) * 64:((pc * 68) + kk) * 64 + 1]),
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
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                        ) as scope_4:
                            value_25 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_26 = T.tensor_view(
                                T.ptr_of(value_25),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_27 = T.tensor_view(
                                T.ptr_of(value_26),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_28 = T.tensor_view(
                                T.ptr_of(value_27),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_view = T.tensor_view(
                                T.ptr_of(value_28[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            value_29 = T.tensor_view(
                                T.ptr_of(out), layout=(16384, 256), shape=(16384, 256)
                            )
                            value_30 = T.tensor_view(
                                T.ptr_of(value_29), layout=(16384, 256), shape=(16384, 256)
                            )
                            value_31 = T.tensor_view(
                                T.ptr_of(value_30), layout=(16384, 256), shape=(16384, 256)
                            )
                            value_32 = T.tensor_view(
                                T.ptr_of(value_31), layout=(16384, 256), shape=(16384, 256)
                            )
                            value_33 = T.tensor_view(
                                T.ptr_of(value_32), layout=(16384, 256), shape=(16384, 256)
                            )
                            window = T.tensor_view(
                                T.ptr_of(value_33[(((((bb - 16) * 8) + mi) * 4) + pc) * 128:(((((bb - 16) * 8) + mi) * 4) + pc) * 128 + 128, 0:0 + 256]),
                                layout=(128, 256),
                                shape=(128, 256),
                            )
                            T.copy(value_view, window)
