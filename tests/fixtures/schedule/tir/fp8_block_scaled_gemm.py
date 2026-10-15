# analysis target=nvidia.h200_sxm module=FP8_BLOCK_SCALED_GEMM function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r130.02KB/w64.00KB@logical,r130.02KB/w64.00KB@total,r130.02KB/w64.00KB@cta,r128.05KB/w32.12KB@thread;rmem:r1.35MB/w1.16MB@logical,r1.35MB/w1.35MB@total,r1.35MB/w1.35MB@cta,r5.17KB/w4.92KB@thread;smem:r128.00KB/w128.00KB@logical,r128.00KB/w128.00KB@total,r128.00KB/w128.00KB@cta,r17.00KB/w128.00KB@thread footprint=a:16.00KB;a_scale:512B;b:16.00KB;b_scale:4B;v24:110:32.00KB footprint-precision=exact peak=gmem:162.02KB;rmem:64.50KB;smem:64.00KB persistent=gmem:130.02KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 512), "fp8e4m3"], b: Tensor[(512, 128), "fp8e4m3", ((512, 128), (1, 512))], a_scale: Tensor[(128, 4), "f32"], b_scale: Tensor[(4, 1), "f32"], out: Tensor[(128, 128), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
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
                        ), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
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
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 128, kb * 128:kb * 128 + 128]),
                        layout=((128, 128), (512, 1)),
                        shape=(128, 128),
                    )
                    T.copy_async_tensor(tile, lhs_ring_stages[kb % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[kb * 128:kb * 128 + 128, 0:0 + 128]),
                        layout=((128, 128), (1, 512)),
                        shape=(128, 128),
                    )
                    T.copy_async_tensor(tile_1, rhs_ring_stages[kb % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    part = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 16, 4 @ scope_3.d3, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(part, 0.0)
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 128, 128):
                            for o_k in range(0, 128, 128):
                                lhs_view = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[kb % 2][o_m:o_m + 128, o_k:o_k + 32]),
                                    layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 32),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[kb % 2][o_k:o_k + 32, o_n:o_n + 128]),
                                    layout=(Layout((32, (16, 8)), (1, (1024, 128))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(32, 128),
                                )
                                T.tiled_mma(
                                    part,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[kb % 2][o_m:o_m + 128, o_k + 32:o_k + 32 + 32]),
                                    layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 32),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[kb % 2][o_k + 32:o_k + 32 + 32, o_n:o_n + 128]),
                                    layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(32, 128),
                                )
                                T.tiled_mma(
                                    part,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[kb % 2][o_m:o_m + 128, o_k + 64:o_k + 64 + 32]),
                                    layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 64 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 32),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[kb % 2][o_k + 64:o_k + 64 + 32, o_n:o_n + 128]),
                                    layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 64 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(32, 128),
                                )
                                T.tiled_mma(
                                    part,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[kb % 2][o_m:o_m + 128, o_k + 96:o_k + 96 + 32]),
                                    layout=(Layout(((2, 8, 8), 32), ((8192, 1024, 128), 1)) + 96 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 32),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[kb % 2][o_k + 96:o_k + 96 + 32, o_n:o_n + 128]),
                                    layout=(Layout((32, (16, 8)), (1, (1024, 128))) + 96 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(32, 128),
                                )
                                T.tiled_mma(
                                    part,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    row_scale_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_3.d0, 8 @ scope_3.d2, 2, 4 @ scope_3.d1), (64, 1, 8, 16)),
                            "rmem",
                        ]
                    )
                    tile_2 = T.tensor_view(
                        T.ptr_of(a_scale[0:0 + 128, kb:kb + 1]),
                        layout=((128, 1), (4, 1)),
                        shape=(128, 1),
                    )
                    T.copy(tile_2, row_scale_dst)
                    value = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 16, 4 @ scope_3.d3, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(part, row_scale_dst, value, kind=BinaryKind.MUL)
                    tile_scale_dst = T.alloc_tensor(
                        tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"]
                    )
                    tile_3 = T.tensor_view(
                        T.ptr_of(b_scale[kb:kb + 1, 0:0 + 1]), layout=(1, 1), shape=(1, 1)
                    )
                    T.copy(tile_3, tile_scale_dst)
                    scaled = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 128),
                            "f32",
                            ((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 16, 4 @ scope_3.d3, 2), (8192, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value, tile_scale_dst, scaled, kind=BinaryKind.MUL)
                    T.binary(acc, scaled, acc, kind=BinaryKind.ADD)
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 128),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 16, 4 @ scope_4.d3, 2), (8192, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(acc, result, dtype="bf16")
                window = T.tensor_view(
                    T.ptr_of(out[0:0 + 128, 0:0 + 128]), layout=(128, 128), shape=(128, 128)
                )
                T.copy(result, window)
