# analysis target=nvidia.h200_sxm module=SCHED_O_PROJ_T1024 function=gemm topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r6.91MB/w10.16MB@logical,r911.62MB/w1.31GB@total,r6.91MB/w10.16MB@cta,r6.83MB/w10.16MB@thread;rmem:r15.87MB/w15.78MB@logical,r2.05GB/w2.04GB@total,r15.87MB/w15.86MB@cta,r69.62KB/w63.44KB@thread;smem:r6.83MB/w6.83MB@logical,r901.31MB/w901.31MB@total,r6.83MB/w6.83MB@cta,r1.06MB/w6.75MB@thread footprint=a:128.00KB;residual:5.16MB;v27:76:40.00KB;v29:77:5.16MB;w:340.00KB footprint-precision=exact peak=gmem:92.00MB;rmem:200.00KB;smem:184.00KB persistent=gmem:82.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(1024, 6144), "bf16"], w: Tensor[(5120, 6144), "bf16"], residual: Tensor[(1024, 5120), "bf16"], out: Tensor[(1024, 5120), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "f32",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "f32",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "f32",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "f32",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        mixed = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 160),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 20), (10240, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            value = T.tensor_view(
                147456, dtype="bf16", storage="smem", layout=(128, 160), shape=(128, 160)
            )
            for u in range(cta.d0, 256, 132):
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as threads:
                    T.fill(acc, 0.0)
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
                for k in range(0, 6144, 64):
                    with scope[:1, :32] as scope_1:
                        tile = T.tensor_view(
                            T.ptr_of(a[(u % 8) * 128:(u % 8) * 128 + 128, k:k + 64]),
                            layout=((128, 64), (6144, 1)),
                            shape=(128, 64),
                        )
                        with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                            T.copy_async_tensor(tile, lhs_ring_stages[(k // 64) % 4])
                        tile_1 = T.tensor_view(
                            T.ptr_of(w[(u // 8) * 160:(u // 8) * 160 + 160, k:k + 64]),
                            layout=((160, 64), (6144, 1)),
                            shape=(160, 64),
                        )
                        with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                            T.copy_async_tensor(tile_1, rhs_ring_stages[(k // 64) % 4])
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_2:
                        tile_2 = T.tensor_view(
                            T.ptr_of(rhs_ring_stages[(k // 64) % 4][0:0 + 160, 0:0 + 64]),
                            layout=Layout(((4, 16), (20, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                            shape=(64, 160),
                        )
                        with Mesh(
                            scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                        ) as threads_3:
                            for o_m in range(0, 64, 64):
                                for o_n in range(0, 160, 160):
                                    for o_k in range(0, 64, 64):
                                        acc_view = T.tensor_view(
                                            T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 160]),
                                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k:o_k + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view = T.tensor_view(
                                            T.ptr_of(tile_2[o_k:o_k + 16, o_n:o_n + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view,
                                            lhs_view,
                                            rhs_view,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                        )
                                        acc_view_1 = T.tensor_view(
                                            T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 160]),
                                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_1 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_1 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k + 16:o_k + 16 + 16, o_n:o_n + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_1,
                                            lhs_view_1,
                                            rhs_view_1,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                        )
                                        acc_view_2 = T.tensor_view(
                                            T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 160]),
                                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_2 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_2 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k + 32:o_k + 32 + 16, o_n:o_n + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_2,
                                            lhs_view_2,
                                            rhs_view_2,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                        )
                                        acc_view_3 = T.tensor_view(
                                            T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 160]),
                                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_3 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_3 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k + 48:o_k + 48 + 16, o_n:o_n + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_3,
                                            lhs_view_3,
                                            rhs_view_3,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                        )
                        with Mesh(
                            scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                        ) as threads_4:
                            for o_m_1 in range(64, 128, 64):
                                for o_n_1 in range(0, 160, 160):
                                    for o_k_1 in range(0, 64, 64):
                                        acc_view_4 = T.tensor_view(
                                            T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 160]),
                                            layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_4 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_4 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_4,
                                            lhs_view_4,
                                            rhs_view_4,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                        )
                                        acc_view_5 = T.tensor_view(
                                            T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 160]),
                                            layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_5 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_5 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_5,
                                            lhs_view_5,
                                            rhs_view_5,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                        )
                                        acc_view_6 = T.tensor_view(
                                            T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 160]),
                                            layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_6 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_6 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_6,
                                            lhs_view_6,
                                            rhs_view_6,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                        )
                                        acc_view_7 = T.tensor_view(
                                            T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 160]),
                                            layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 20), (1, 8, 16, 64, 128, 512)),
                                            shape=(64, 160),
                                        )
                                        lhs_view_7 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                            layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view_7 = T.tensor_view(
                                            T.ptr_of(tile_2[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 160]),
                                            layout=(Layout((16, (20, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 160),
                                        )
                                        T.tiled_mma(
                                            acc_view_7,
                                            lhs_view_7,
                                            rhs_view_7,
                                            atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                        )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    tile_3 = T.tensor_view(
                        T.ptr_of(residual[(u % 8) * 128:(u % 8) * 128 + 128, (u // 8) * 160:(u // 8) * 160 + 160]),
                        layout=((128, 160), (5120, 1)),
                        shape=(128, 160),
                    )
                    with scope_3 as threads_5:
                        dst_frame = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.copy(tile_3, dst_frame)
                        src_frame = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.cast(src_frame, dst_frame_1, dtype="f32")
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(mixed[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.cast(src_frame_1, dst_frame_2, dtype="bf16")
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(mixed[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.cast(src_frame_2, dst_frame_3, dtype="f32")
                        lhs_frame = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.binary(lhs_frame, rhs_frame, dst_frame_4, kind=BinaryKind.ADD)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(tile_out[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.cast(src_frame_3, dst_frame_5, dtype="bf16")
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(tile_out[0:0 + 128, 0:0 + 160]),
                            layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 20), (10240, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 160),
                        )
                        T.copy(src_frame_4, value)
                with scope[1:2, :32] as scope_4:
                    with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_12:
                        window = T.tensor_view(
                            T.ptr_of(out[((128 * u) + (1024 * (((0 - u) - 1) // 8))) + 1024:((128 * u) + (1024 * (((0 - u) - 1) // 8))) + 1024 + 128, (-160 * (((0 - u) - 1) // 8)) - 160:(-160 * (((0 - u) - 1) // 8)) - 160 + 160]),
                            layout=((128, 160), (5120, 1)),
                            shape=(128, 160),
                        )
                        T.copy_async_tensor(value, window)
