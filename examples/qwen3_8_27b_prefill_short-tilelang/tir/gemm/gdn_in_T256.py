# analysis target=nvidia.h200_sxm module=SCHED_GDN_IN_T256 function=gemm topology=cta wave=130/130
# selection requested=memory executed=memory
# memory traffic=gmem:r3.81MB/w8.25MB@logical,r495.62MB/w1.05GB@total,r3.81MB/w8.25MB@cta,r3.81MB/w8.25MB@thread;rmem:r10.19MB/w10.19MB@logical,r1.29GB/w1.29GB@total,r10.19MB/w10.19MB@cta,r43.30KB/w40.77KB@thread;smem:r3.81MB/w3.81MB@logical,r495.62MB/w495.62MB@total,r3.81MB/w3.81MB@cta,r724.00KB/w3.75MB@thread footprint=a:16.00KB;v21:56:64.00KB;w:32.00KB footprint-precision=lower_bound peak=gmem:181.31MB;rmem:128.02KB;smem:192.00KB persistent=gmem:165.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 5120), "bf16"], w: Tensor[(16640, 5120), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(256, 16640), "bf16"]
):
    with Mesh((Topology("cta", 130),), layout=(65, 2), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        staged_dst = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(128, 256), shape=(128, 256)
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_dst_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for k in range(0, 5120, 64):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 128, k:k + 64]),
                        layout=((128, 64), (5120, 1)),
                        shape=(128, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_ring_stages[(k // 64) % 4])
                    tile_1 = T.tensor_view(
                        T.ptr_of(w[cta.d0 * 256:cta.d0 * 256 + 256, k:k + 64]),
                        layout=((256, 64), (5120, 1)),
                        shape=(256, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_ring_stages[(k // 64) % 4])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_2:
                    tile_2 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 4][0:0 + 256, 0:0 + 64]),
                        layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(64, 256),
                    )
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
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_2[o_k:o_k + 16, o_n:o_n + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
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
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                        layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
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
                    T.cast(src_frame, dst_frame, dtype="bf16")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(tile_out[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.copy(src_frame_1, staged_dst)
            with scope[:1, :32] as scope_4:
                with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_7:
                    T.copy_async_tensor(staged_dst, tile_dst_1)
                window = T.tensor_view(
                    T.ptr_of(out[cta.d1 * 128:cta.d1 * 128 + 128, cta.d0 * 256:cta.d0 * 256 + 256]),
                    layout=((128, 256), (16640, 1)),
                    shape=(128, 256),
                )
                T.copy(tile_dst_1, window)
