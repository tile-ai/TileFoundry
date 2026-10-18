# analysis target=nvidia.h200_sxm module=PersistentTileUnionCount function=gemm topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r1.41MB/w67.12MB@logical,r180.00MB/w8.39GB@total,r1.41MB/w67.12MB@cta,r1.41MB/w67.12MB@thread;rmem:r3.94MB/w2.94MB@logical,r504.33MB/w504.00MB@total,r3.94MB/w3.94MB@cta,r18.42KB/w15.75KB@thread;smem:r1.41MB/w1.41MB@logical,r180.00MB/w180.00MB@total,r1.41MB/w1.41MB@cta,r724.50KB/w866.25KB@thread footprint=a:2.00MB;b:128.00KB;v24:98:64.00KB;v28:100:8.00MB footprint-precision=exact peak=gmem:72.19MB;rmem:128.00KB;smem:160.00KB persistent=gmem:6.12MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(4224, 512), "bf16"], b: Tensor[(512, 2048), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(16896, 2048), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(128,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for work in range(cta.block, 1056, 128):
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
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
                for kk in range(0, 2, 1):
                    with scope[:1, :32] as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(a[((work // 4) % 33) * 128:((work // 4) % 33) * 128 + 1, (((work % 4) * 2) + kk) * 64:(((work % 4) * 2) + kk) * 64 + 1]),
                            layout=((128, 64), (512, 1)),
                            shape=(128, 64),
                        )
                        T.copy_async_tensor(
                            value,
                            lhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2],
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(b[(((work % 4) * 2) + kk) * 64:(((work % 4) * 2) + kk) * 64 + 1, ((work // 4) // 33) * 256:((work // 4) // 33) * 256 + 1]),
                            layout=((64, 256), (2048, 1)),
                            shape=(64, 256),
                        )
                        T.copy_async_tensor(
                            value_1,
                            rhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2],
                        )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                    ) as scope_3:
                        acc_view = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        lhs_view = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][0:0 + 128, 0:0 + 16]),
                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(128, 16),
                        )
                        rhs_view = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][0:0 + 16, 0:0 + 256]),
                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 256),
                        )
                        T.tiled_mma(
                            acc_view,
                            lhs_view,
                            rhs_view,
                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
                        acc_view_1 = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        lhs_view_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][0:0 + 128, 16:16 + 16]),
                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                            shape=(128, 16),
                        )
                        rhs_view_1 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][16:16 + 16, 0:0 + 256]),
                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 256),
                        )
                        T.tiled_mma(
                            acc_view_1,
                            lhs_view_1,
                            rhs_view_1,
                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
                        acc_view_2 = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        lhs_view_2 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][0:0 + 128, 32:32 + 16]),
                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                            shape=(128, 16),
                        )
                        rhs_view_2 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][32:32 + 16, 0:0 + 256]),
                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 256),
                        )
                        T.tiled_mma(
                            acc_view_2,
                            lhs_view_2,
                            rhs_view_2,
                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
                        acc_view_3 = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 32, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        lhs_view_3 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][0:0 + 128, 48:48 + 16]),
                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                            shape=(128, 16),
                        )
                        rhs_view_3 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((work - cta.block) // 128) * 2) + kk) % 2][48:48 + 16, 0:0 + 256]),
                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                            shape=(16, 256),
                        )
                        T.tiled_mma(
                            acc_view_3,
                            lhs_view_3,
                            rhs_view_3,
                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                        )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_4:
                    tile_out = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 256),
                            "bf16",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    src_view = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    T.cast(src_view, tile_out, dtype="bf16")
                    staged_dst = T.tensor_view(
                        98304,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    T.copy(tile_out, staged_dst)
                with scope[:1, :32] as scope_5:
                    staged_dst_2 = T.tensor_view(
                        98304,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    tile_dst_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
                    )
                    T.copy_async_tensor(staged_dst_2, tile_dst_1)
                    window = T.tensor_view(
                        T.ptr_of(out[((work % 4) * 4224) + (((work // 4) % 33) * 128):((work % 4) * 4224) + (((work // 4) % 33) * 128) + 128, ((work // 4) // 33) * 256:((work // 4) // 33) * 256 + 256]),
                        layout=((128, 256), (2048, 1)),
                        shape=(128, 256),
                    )
                    T.copy(tile_dst_1, window)
