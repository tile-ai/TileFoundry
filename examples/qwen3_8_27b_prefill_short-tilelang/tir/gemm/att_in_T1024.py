# analysis target=nvidia.h200_sxm module=SCHED_ATT_IN_T1024 function=gemm topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r13.97MB/w28.44MB@logical,r1.80GB/w3.67GB@total,r13.97MB/w28.44MB@cta,r13.97MB/w28.44MB@thread;rmem:r35.67MB/w35.33MB@logical,r4.60GB/w4.60GB@total,r35.67MB/w35.66MB@cta,r152.94KB/w142.62KB@thread;smem:r13.97MB/w13.97MB@logical,r1.80GB/w1.80GB@total,r13.97MB/w13.97MB@cta,r2.48MB/w13.75MB@thread footprint=a:128.00KB;v20:71:56.00KB;v22:72:7.22MB;w:476.00KB footprint-precision=exact peak=gmem:178.00MB;rmem:112.00KB;smem:188.00KB persistent=gmem:150.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(1024, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], out: Tensor[(1024, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(86016, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(102400, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(118784, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((28, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(224, 64)),
                T.tensor_view(28672, dtype="bf16", storage="smem", layout=Layout(((28, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(224, 64)),
                T.tensor_view(57344, dtype="bf16", storage="smem", layout=Layout(((28, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(224, 64)),
            )
            for u in range(cta.d0, 512, 132):
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 224),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 28, 4, 2), (14336, 16, 8, 1, 512, 128, 64)),
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
                            T.ptr_of(a[(u % 8) * 128:(u % 8) * 128 + 1, k:k + 1]),
                            layout=((128, 64), (5120, 1)),
                            shape=(128, 64),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 64),
                        )
                        T.copy_async_tensor(value, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(w[(u // 8) * 224:(u // 8) * 224 + 1, k:k + 1]),
                            layout=((224, 64), (5120, 1)),
                            shape=(224, 64),
                        )
                        value_3 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                            layout=Layout(((28, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(224, 64),
                        )
                        T.copy_async_tensor(value_2, value_3)
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        for o_m in range(0, 128, 128):
                            for o_n in range(0, 224, 224):
                                for o_k in range(0, 64, 64):
                                    value_4 = T.tensor_view(
                                        T.ptr_of(acc),
                                        layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 28, 4 @ scope_3.d3, 2), (14336, 16, 8, 1, 512, 128, 64)),
                                        shape=(128, 224),
                                    )
                                    value_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    value_6 = T.tensor_view(
                                        T.ptr_of(rhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=((64, 224), (1, 64)) | Swizzle(3, 4, 3),
                                        shape=(64, 224),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 224]),
                                        layout=(Layout((16, (28, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 224),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=224, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 224]),
                                        layout=(Layout((16, (28, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 224),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=224, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 224]),
                                        layout=(Layout((16, (28, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 224),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=224, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 224]),
                                        layout=(Layout((16, (28, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 224),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=224, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    tile_out = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 224),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 28, 4 @ scope_4.d3, 2), (14336, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 28, 4 @ scope_4.d3, 2), (14336, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 224),
                    )
                    T.cast(value_8, tile_out, dtype="bf16")
                    value_7 = T.tensor_view(
                        135168,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 224),
                        shape=(128, 224),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_7), layout=(128, 224), shape=(128, 224)
                    )
                    T.copy(tile_out, value_9)
                with scope[1:2, :32] as scope_5:
                    value_7_2 = T.tensor_view(
                        135168,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 224),
                        shape=(128, 224),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_7_2), layout=(128, 224), shape=(128, 224)
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(out[(128 * u) - (1024 * (u // 8)):(128 * u) - (1024 * (u // 8)) + 1, (224 * u) - (224 * (((7 * u) + 7) // 8)):(224 * u) - (224 * (((7 * u) + 7) // 8)) + 1]),
                        layout=((128, 224), (14336, 1)),
                        shape=(128, 224),
                    )
                    T.copy_async_tensor(value_10, value_11)
