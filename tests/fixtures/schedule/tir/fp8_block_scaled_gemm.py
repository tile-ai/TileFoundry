# analysis target=nvidia.h200_sxm module=FP8_BLOCK_SCALED_GEMM function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r130.02KB/w64.00KB@logical,r130.02KB/w64.00KB@total,r130.02KB/w64.00KB@cta,r128.05KB/w32.12KB@thread;rmem:r1.35MB/w1.16MB@logical,r1.35MB/w1.35MB@total,r1.35MB/w1.35MB@cta,r5.17KB/w4.92KB@thread;smem:r128.00KB/w128.00KB@logical,r128.00KB/w128.00KB@total,r128.00KB/w128.00KB@cta,r17.00KB/w128.00KB@thread footprint=a:16.00KB;a_scale:512B;b:16.00KB;b_scale:4B;v24:110:32.00KB footprint-precision=exact peak=gmem:162.02KB;rmem:128.50KB;smem:64.00KB persistent=gmem:130.02KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 512), "fp8e4m3"], b: Tensor[(512, 128), "fp8e4m3", ((512, 128), (1, 512))], a_scale: Tensor[(128, 4), "f32"], b_scale: Tensor[(4, 1), "f32"], out: Tensor[(128, 128), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 128),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 16, 4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(1), S(3), S(5)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                        ), names=("group", "warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_1:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(32768, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
                T.tensor_view(49152, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="fp8e4m3", storage="smem", layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3), shape=(128, 128)),
                T.tensor_view(16384, dtype="fp8e4m3", storage="smem", layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3), shape=(128, 128)),
            )
            for kb in range(0, 4, 1):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(T.ptr_of(a), layout=(128, 512), shape=(128, 512))
                    value_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 1, kb * 128:kb * 128 + 1]),
                        layout=((128, 128), (512, 1)),
                        shape=(128, 128),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[kb % 2]),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    T.copy_async_tensor(value_1, value_2)
                    value_3 = T.tensor_view(
                        T.ptr_of(b), layout=((512, 128), (1, 512)), shape=(512, 128)
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[kb * 128:kb * 128 + 1, 0:0 + 1]),
                        layout=((128, 128), (1, 512)),
                        shape=(128, 128),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[kb % 2]),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_3:
                    part = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(part, 0.0)
                    acc_view = T.tensor_view(
                        T.ptr_of(part[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    value_6 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[kb % 2]),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_8),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 128, 0:0 + 32]),
                        layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 32),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[kb % 2]),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_13),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 32, 0:0 + 128]),
                        layout=(Layout((32, (16, 8)), (1, (1024, 128))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(32, 128),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(part[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 128, 32:32 + 32]),
                        layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 32),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_15[32:32 + 32, 0:0 + 128]),
                        layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(32, 128),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        lhs_view_1,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_2 = T.tensor_view(
                        T.ptr_of(part[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    lhs_view_2 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 128, 64:64 + 32]),
                        layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 64 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 32),
                    )
                    rhs_view_2 = T.tensor_view(
                        T.ptr_of(value_15[64:64 + 32, 0:0 + 128]),
                        layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 64 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(32, 128),
                    )
                    T.tiled_mma(
                        acc_view_2,
                        lhs_view_2,
                        rhs_view_2,
                        atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_3 = T.tensor_view(
                        T.ptr_of(part[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    lhs_view_3 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 128, 96:96 + 32]),
                        layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 96 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 32),
                    )
                    rhs_view_3 = T.tensor_view(
                        T.ptr_of(value_15[96:96 + 32, 0:0 + 128]),
                        layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 96 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(32, 128),
                    )
                    T.tiled_mma(
                        acc_view_3,
                        lhs_view_3,
                        rhs_view_3,
                        atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    row_scale_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_3.group, 8 @ scope_3.lane8, 2, 4 @ scope_3.warp), (64, 1, 8, 16)),
                            "rmem",
                        ]
                    )
                    value_16 = T.tensor_view(T.ptr_of(a_scale), layout=(128, 4), shape=(128, 4))
                    value_17 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 1, kb:kb + 1]),
                        layout=((128, 1), (4, 0)),
                        shape=(128, 1),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(row_scale_dst),
                        layout=((2 @ scope_3.group, 8 @ scope_3.lane8, 2, 4 @ scope_3.warp), (64, 1, 8, 16)),
                        shape=(128, 1),
                    )
                    T.copy(value_17, value_18)
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_20 = T.tensor_view(
                        T.ptr_of(part),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    lhs_view_4 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    T.binary(lhs_view_4, value_18, value_19, kind=BinaryKind.MUL)
                    tile_scale_dst = T.alloc_tensor(
                        tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"]
                    )
                    value_21 = T.tensor_view(T.ptr_of(b_scale), layout=(4, 1), shape=(4, 1))
                    value_22 = T.tensor_view(
                        T.ptr_of(value_21[kb:kb + 1, 0:0 + 1]),
                        layout=((1, 1), (0, 0)),
                        shape=(1, 1),
                    )
                    value_23 = T.tensor_view(
                        T.ptr_of(tile_scale_dst), layout=(1, 1), shape=(1, 1)
                    )
                    T.copy(value_22, value_23)
                    scaled = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_19, value_23, scaled, kind=BinaryKind.MUL)
                    value_24 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    value_25 = T.tensor_view(
                        T.ptr_of(value_24),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    value_26 = T.tensor_view(
                        T.ptr_of(value_25),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    value_27 = T.tensor_view(
                        T.ptr_of(value_26),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    lhs_view_5 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    dst_view = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 16, 4 @ scope_3.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 128),
                    )
                    T.binary(lhs_view_5, scaled, dst_view, kind=BinaryKind.ADD)
            value_28 = T.tensor_view(T.ptr_of(out), layout=(128, 128), shape=(128, 128))
            T.fill(value_28, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 128),
                        "bf16",
                        ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_29 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 128),
                )
                value_30 = T.tensor_view(
                    T.ptr_of(value_29),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 128),
                )
                value_31 = T.tensor_view(
                    T.ptr_of(value_30),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 128),
                )
                value_32 = T.tensor_view(
                    T.ptr_of(value_31),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 128),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_32[0:0 + 128, 0:0 + 128]),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 16, 4 @ scope_4.lane4, 2), (8192, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 128),
                )
                T.cast(src_view, result, dtype="bf16")
                value_33 = T.tensor_view(T.ptr_of(out), layout=(128, 128), shape=(128, 128))
                window = T.tensor_view(
                    T.ptr_of(value_33[0:0 + 128, 0:0 + 128]),
                    layout=(128, 128),
                    shape=(128, 128),
                )
                T.copy(result, window)
