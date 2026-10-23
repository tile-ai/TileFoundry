from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for m in range(0, 8192, 128):
                for n in range(0, 17408, 256):
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
                                ), names=("d0", "d1", "d2", "d3")),
                            ),
                            "rmem",
                        ]
                    )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_1:
                        T.fill(acc, 0.0)
                    for k in range(0, 5120, 64):
                        with scope[:1, :32] as scope_2:
                            value = T.tensor_view(
                                T.ptr_of(a[m:m + 1, k:k + 1]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            value_1 = T.tensor_view(
                                T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value, value_1)
                            value_2 = T.tensor_view(
                                T.ptr_of(b[k:k + 1, n:n + 1]),
                                layout=((64, 256), (17408, 1)),
                                shape=(64, 256),
                            )
                            value_3 = T.tensor_view(
                                T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3]),
                                layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            T.copy_async_tensor(value_2, value_3)
                        with Mesh(
                            scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_3:
                            for o_m in range(0, 128, 128):
                                for o_n in range(0, 256, 256):
                                    for o_k in range(0, 64, 64):
                                        value_4 = T.tensor_view(
                                            T.ptr_of(acc),
                                            layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                            shape=(128, 256),
                                        )
                                        value_5 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3]),
                                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                            shape=(128, 64),
                                        )
                                        lhs_view = T.tensor_view(
                                            T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        value_6 = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3]),
                                            layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                            shape=(64, 256),
                                        )
                                        rhs_view = T.tensor_view(
                                            T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_4,
                                            lhs_view,
                                            rhs_view,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_1 = T.tensor_view(
                                            T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_1 = T.tensor_view(
                                            T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_4,
                                            lhs_view_1,
                                            rhs_view_1,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_2 = T.tensor_view(
                                            T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_2 = T.tensor_view(
                                            T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_4,
                                            lhs_view_2,
                                            rhs_view_2,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_3 = T.tensor_view(
                                            T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_3 = T.tensor_view(
                                            T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_4,
                                            lhs_view_3,
                                            rhs_view_3,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_4:
                        tile_out = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_7 = T.tensor_view(
                            T.ptr_of(acc),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(value_7, tile_out, dtype="bf16")
                        staged_dst = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_8 = T.tensor_view(
                            T.ptr_of(staged_dst),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(tile_out, value_8)
                    with scope[:1, :32] as scope_5:
                        staged_dst_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_9 = T.tensor_view(
                            T.ptr_of(staged_dst_2),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_10 = T.tensor_view(
                            T.ptr_of(out[m:m + 1, n:n + 1]),
                            layout=((128, 256), (17408, 1)),
                            shape=(128, 256),
                        )
                        T.copy_async_tensor(value_9, value_10)
