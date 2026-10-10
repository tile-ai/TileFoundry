# analysis target=nvidia.h200_sxm module=WGMMA_REPEAT_ALONG_K function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r36.00KB/w0@logical,r36.00KB/w0@total,r36.00KB/w0@cta,r36.00KB/w0@thread;rmem:r24.06KB/w28.00KB@logical,r24.06KB/w28.00KB@total,r24.06KB/w28.00KB@cta,r160B/w112B@thread;smem:r36.00KB/w36.00KB@logical,r36.00KB/w36.00KB@total,r36.00KB/w36.00KB@cta,r1.50KB/w36.00KB@thread footprint=a:16.00KB;b:2.00KB footprint-precision=exact peak=gmem:36.00KB;rmem:8.00KB;smem:36.00KB persistent=gmem:36.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 128), "bf16"], b: Tensor[(128, 16), "bf16"], out: Tensor[(128, 16), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 16),
                "f32",
                ((2, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 16),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_stages = (
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 8)), ((256, 64, 8), (128, 1))), shape=(64, 16)),
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 8)), ((256, 64, 8), (128, 1))), shape=(64, 16)),
            )
            for k in range(0, 128, 64):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 128, k:k + 64]),
                        layout=((128, 64), (128, 1)),
                        shape=(128, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 64) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 64, 0:0 + 16]), layout=(64, 16), shape=(64, 16)
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 64) % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_2:
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_3:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 16, 16):
                                for o_k in range(0, 64, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k:o_k + 16, o_n:o_n + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=16,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k + 16:o_k + 16 + 16, o_n:o_n + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k + 32:o_k + 32 + 16, o_n:o_n + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=48,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k + 48:o_k + 48 + 16, o_n:o_n + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                    )
                    with Mesh(
                        scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_4:
                        for o_m_1 in range(64, 128, 64):
                            for o_n_1 in range(0, 16, 16):
                                for o_k_1 in range(0, 64, 64):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=16,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=48,
                                                outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 64) % 2][o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                    )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_3:
                with scope_3 as threads_5:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 16),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(result[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 16),
                    )
                    T.cast(src_frame, dst_frame, dtype="bf16")
        with Mesh(
            (Topology("thread", 384),), layout=(2, 4, 8, 4) + 128, names=("d0", "d1", "d2", "d3")
        ) as threads_6:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 128, 0:0 + 16]),
                layout=((2 @ threads_6.d0, 8 @ threads_6.d2, 2, 4 @ threads_6.d1, 2, 4 @ threads_6.d3, 2), (1024, 1, 8, 16, 64, 128, 512)),
                shape=(128, 16),
            )
            T.copy(result_view, out)
