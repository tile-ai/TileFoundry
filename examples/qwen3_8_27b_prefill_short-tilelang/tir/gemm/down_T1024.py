# analysis target=nvidia.h200_sxm module=SCHED_DOWN_T1024 function=gemm topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r19.28MB/w10.16MB@logical,r2.49GB/w1.31GB@total,r19.28MB/w10.16MB@cta,r19.20MB/w10.16MB@thread;rmem:r43.38MB/w43.28MB@logical,r5.59GB/w5.59GB@total,r43.38MB/w43.36MB@cta,r190.62KB/w173.44KB@thread;smem:r19.20MB/w19.20MB@logical,r2.48GB/w2.48GB@total,r19.20MB/w19.20MB@cta,r2.87MB/w19.13MB@thread footprint=a:128.00KB;residual:5.16MB;v27:76:40.00KB;v29:77:5.16MB;w:340.00KB footprint-precision=exact peak=gmem:224.00MB;rmem:200.00KB;smem:184.00KB persistent=gmem:214.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(1024, 17408), "bf16"], w: Tensor[(5120, 17408), "bf16"], residual: Tensor[(1024, 5120), "bf16"], out: Tensor[(1024, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(40960, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(61440, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
            )
            for u in range(cta.d0, 256, 132):
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 160),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 20, 4, 2), (10240, 16, 8, 1, 512, 128, 64)),
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
                for k in range(0, 17408, 64):
                    with scope[:1, :32] as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(a[(u % 8) * 128:(u % 8) * 128 + 1, k:k + 1]),
                            layout=((128, 64), (17408, 1)),
                            shape=(128, 64),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 272) + (k // 64)) % 4]),
                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 64),
                        )
                        T.copy_async_tensor(value, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(w[(u // 8) * 160:(u // 8) * 160 + 1, k:k + 1]),
                            layout=((160, 64), (17408, 1)),
                            shape=(160, 64),
                        )
                        value_3 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((u - cta.d0) // 132) * 272) + (k // 64)) % 4]),
                            layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(160, 64),
                        )
                        T.copy_async_tensor(value_2, value_3)
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        for o_m in range(0, 128, 128):
                            for o_n in range(0, 160, 160):
                                for o_k in range(0, 64, 64):
                                    value_4 = T.tensor_view(
                                        T.ptr_of(acc),
                                        layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 20, 4 @ scope_3.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                                        shape=(128, 160),
                                    )
                                    value_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 272) + (k // 64)) % 4]),
                                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    value_6 = T.tensor_view(
                                        T.ptr_of(rhs_ring_stages[((((u - cta.d0) // 132) * 272) + (k // 64)) % 4]),
                                        layout=((64, 160), (1, 64)) | Swizzle(3, 4, 3),
                                        shape=(64, 160),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 160]),
                                        layout=(Layout((16, (20, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 160),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 160]),
                                        layout=(Layout((16, (20, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 160),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 160]),
                                        layout=(Layout((16, (20, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 160),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 160]),
                                        layout=(Layout((16, (20, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 160),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    value_8 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(residual[(u % 8) * 128:(u % 8) * 128 + 1, (u // 8) * 160:(u // 8) * 160 + 1]),
                        layout=((128, 160), (5120, 1)),
                        shape=(128, 160),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_8),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 160),
                    )
                    T.copy(value_9, value_10)
                    value_11 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_10, value_11, dtype="f32")
                    mixed = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 160),
                    )
                    T.cast(value_12, mixed, dtype="bf16")
                    value_13 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(mixed, value_13, dtype="f32")
                    value_14 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_11, value_13, value_14, kind=BinaryKind.ADD)
                    tile_out = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 160),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_14, tile_out, dtype="bf16")
                    value_7 = T.tensor_view(
                        147456,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 160),
                        shape=(128, 160),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_7), layout=(128, 160), shape=(128, 160)
                    )
                    T.copy(tile_out, value_15)
                with scope[1:2, :32] as scope_5:
                    value_7_2 = T.tensor_view(
                        147456,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 160),
                        shape=(128, 160),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_7_2), layout=(128, 160), shape=(128, 160)
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(out[(128 * u) - (1024 * (u // 8)):(128 * u) - (1024 * (u // 8)) + 1, (160 * u) - (160 * (((7 * u) + 7) // 8)):(160 * u) - (160 * (((7 * u) + 7) // 8)) + 1]),
                        layout=((128, 160), (5120, 1)),
                        shape=(128, 160),
                    )
                    T.copy_async_tensor(value_16, value_17)
