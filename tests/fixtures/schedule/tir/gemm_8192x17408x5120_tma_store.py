# analysis target=nvidia.h200_sxm module=GEMM_8192X17408X5120_TMA_STORE function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r522.00MB/w816.00MB@logical,r16.20GB/w816.00MB@total,r16.20GB/w816.00MB@cta,r16.20GB/w816.00MB@thread;rmem:r43.30GB/w42.77GB@logical,r43.31GB/w43.30GB@total,r43.31GB/w43.30GB@cta,r183.88MB/w173.19MB@thread;smem:r16.20GB/w522.00MB@logical,r16.20GB/w16.20GB@total,r16.20GB/w16.20GB@cta,r3.00GB/w15.94GB@thread footprint=a:16.00KB;b:32.00KB;v11:71:128.00KB;v12:72:64.00KB footprint-precision=exact peak=gmem:522.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:250.00MB
#   buffer=b holds=175.62MB time=m space=none reuse=10.46GB fits=no precision=exact
#   buffer=a holds=3.94MB time=n space=none reuse=83.75MB fits=yes precision=exact
#   error="l2 reuse window m holds 175.62MB at a 1-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *  # noqa: F401, F403
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        T.fill(out, 0.0)
        with Mesh(
            (Topology("thread", 384),), Layout((3, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            for m in range(0, 8192, 128):
                staged = T.tensor_view(
                    147456,
                    dtype='bf16',
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                for n in range(0, 17408, 256):
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as threads:
                        T.fill(acc, 0.0)
                    lhs_stages = (T.tensor_view(98304, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)), T.tensor_view(114688, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)), T.tensor_view(131072, dtype='bf16', storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)))
                    rhs_stages = (T.tensor_view(0, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)), T.tensor_view(32768, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)), T.tensor_view(65536, dtype='bf16', storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)))
                    for k in range(0, 5120, 64):
                        with scope[:1, :32] as scope_1:
                            tile = T.tensor_view(
                                T.ptr_of(a[m:m + 128, k:k + 64]),
                                layout=Layout((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                                T.copy_async_tensor(tile, lhs_stages[(k // 64) % 3])
                            tile_1 = T.tensor_view(
                                T.ptr_of(b[k:k + 64, n:n + 256]),
                                layout=Layout((64, 256), (17408, 1)),
                                shape=(64, 256),
                            )
                            with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                                T.copy_async_tensor(tile_1, rhs_stages[(k // 64) % 3])
                        with Mesh(
                            scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_2:
                            with Mesh(
                                scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                            ) as threads_3:
                                for o_m in range(0, 64, 64):
                                    for o_n in range(0, 256, 256):
                                        for o_k in range(0, 64, 64):
                                            acc_view = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k:o_k + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k:o_k + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view,
                                                lhs_view,
                                                rhs_view,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                            )
                                            acc_view_1 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_1 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_1,
                                                lhs_view_1,
                                                rhs_view_1,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                            )
                                            acc_view_2 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_2 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_2,
                                                lhs_view_2,
                                                rhs_view_2,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                            )
                                            acc_view_3 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_3 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_3,
                                                lhs_view_3,
                                                rhs_view_3,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                            )
                            with Mesh(
                                scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                            ) as threads_4:
                                for o_m_1 in range(64, 128, 64):
                                    for o_n_1 in range(0, 256, 256):
                                        for o_k_1 in range(0, 64, 64):
                                            acc_view_4 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_4 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_4,
                                                lhs_view_4,
                                                rhs_view_4,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                            )
                                            acc_view_5 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_5 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_5,
                                                lhs_view_5,
                                                rhs_view_5,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                            )
                                            acc_view_6 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_6 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_6,
                                                lhs_view_6,
                                                rhs_view_6,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                            )
                                            acc_view_7 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_7 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
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
                                                T.ptr_of(rhs_stages[(k // 64) % 3][o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_7,
                                                lhs_view_7,
                                                rhs_view_7,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_4),
                                            )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        with scope_3 as threads_5:
                            src_frame = T.tensor_view(
                                T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            dst_frame = T.tensor_view(
                                T.ptr_of(tile_out[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.cast(src_frame, dst_frame, dtype='bf16')
                            src_frame_1 = T.tensor_view(
                                T.ptr_of(tile_out[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.copy(src_frame_1, staged)
                    with scope[:1, :32] as scope_4:
                        with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_7:
                            window = T.tensor_view(
                                T.ptr_of(out[m:m + 128, n:n + 256]),
                                layout=Layout((128, 256), (17408, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(staged, window)
