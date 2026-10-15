# analysis target=nvidia.h200_sxm module=SCHED_O_PROJ_T256 function=gemm topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r2.48MB/w2.54MB@logical,r317.00MB/w325.00MB@total,r2.48MB/w2.54MB@cta,r2.46MB/w2.54MB@thread;rmem:r3.97MB/w3.96MB@logical,r507.88MB/w507.50MB@total,r3.97MB/w3.96MB@cta,r18.92KB/w15.88KB@thread;smem:r2.46MB/w2.46MB@logical,r314.50MB/w314.50MB@total,r2.46MB/w2.46MB@cta,r284.00KB/w2.44MB@thread footprint=a:16.00KB;residual:20.00KB;v28:59:20.00KB;w:10.00KB footprint-precision=lower_bound peak=gmem:70.52MB;rmem:100.02KB;smem:156.00KB persistent=gmem:65.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 6144), "bf16"], w: Tensor[(5120, 6144), "bf16"], residual: Tensor[(256, 5120), "bf16"], scratch: Tensor[(10240,), "bf16"], out: Tensor[(256, 5120), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(64, 2), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        staged_dst = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(128, 80), shape=(128, 80)
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        r_dst = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        mixed = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_dst_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 10240]), layout=(128, 80), shape=(128, 80)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(61440, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(77824, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(94208, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(110592, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(126976, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(143360, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(10240, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(30720, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(40960, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(51200, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
            )
            for k in range(0, 6144, 64):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 128, k:k + 64]),
                        layout=((128, 64), (6144, 1)),
                        shape=(128, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_ring_stages[(k // 64) % 6])
                    tile_1 = T.tensor_view(
                        T.ptr_of(w[cta.d0 * 80:cta.d0 * 80 + 80, k:k + 64]),
                        layout=((80, 64), (6144, 1)),
                        shape=(80, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_ring_stages[(k // 64) % 6])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_2:
                    tile_2 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 6][0:0 + 80, 0:0 + 64]),
                        layout=Layout(((4, 16), (10, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(64, 80),
                    )
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_3:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 80, 80):
                                for o_k in range(0, 64, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_2[o_k:o_k + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 16:o_k + 16 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 32:o_k + 32 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 48:o_k + 48 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                    with Mesh(
                        scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_4:
                        for o_m_1 in range(64, 128, 64):
                            for o_n_1 in range(0, 80, 80):
                                for o_k_1 in range(0, 64, 64):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 6][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_3:
                tile_3 = T.tensor_view(
                    T.ptr_of(residual[cta.d1 * 128:cta.d1 * 128 + 128, cta.d0 * 80:cta.d0 * 80 + 80]),
                    layout=((128, 80), (5120, 1)),
                    shape=(128, 80),
                )
                with scope_3 as threads_5:
                    dst_frame = T.tensor_view(
                        T.ptr_of(r_dst[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.copy(tile_3, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(r_dst[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.cast(src_frame, dst_frame_1, dtype="f32")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(mixed[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.cast(src_frame_1, dst_frame_2, dtype="bf16")
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(mixed[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.cast(src_frame_2, dst_frame_3, dtype="f32")
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.binary(lhs_frame, rhs_frame, dst_frame_4, kind=BinaryKind.ADD)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(tile_out[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.cast(src_frame_3, dst_frame_5, dtype="bf16")
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(tile_out[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 80),
                    )
                    T.copy(src_frame_4, staged_dst)
            with scope[:1, :32] as scope_4:
                with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_12:
                    T.copy_async_tensor(staged_dst, tile_dst_1)
                window = T.tensor_view(
                    T.ptr_of(out[cta.d1 * 128:cta.d1 * 128 + 128, cta.d0 * 80:cta.d0 * 80 + 80]),
                    layout=((128, 80), (5120, 1)),
                    shape=(128, 80),
                )
                T.copy(tile_dst_1, window)
