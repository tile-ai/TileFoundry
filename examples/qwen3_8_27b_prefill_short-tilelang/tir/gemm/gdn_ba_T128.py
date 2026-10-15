# analysis target=nvidia.h200_sxm module=SCHED_GDN_BA_T128 function=gemm topology=cta wave=48/48
# selection requested=memory executed=memory
# memory traffic=gmem:r188.00KB/w400.00KB@logical,r8.81MB/w18.75MB@total,r188.00KB/w400.00KB@cta,r188.00KB/w392.03KB@thread;rmem:r88.37KB/w88.02KB@logical,r4.14MB/w4.13MB@total,r88.37KB/w88.02KB@cta,r728B/w368B@thread;smem:r180.00KB/w180.00KB@logical,r8.44MB/w8.44MB@total,r180.00KB/w180.00KB@cta,r7.50KB/w180.00KB@thread footprint=a:16.00KB;v24:55:8.00KB;w:2.00KB footprint-precision=lower_bound peak=gmem:2.95MB;rmem:8.02KB;smem:36.00KB persistent=gmem:2.19MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 5120), "bf16"], w: Tensor[(96, 5120), "bf16"], scratch: Tensor[(2048,), "f32"], out: Tensor[(8, 128, 96), "f32"]
):
    with Mesh((Topology("cta", 48),), layout=(8, 1, 6), names=("d0", "d1", "d2")) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 16),
                "f32",
                ((2, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        T.fill(out, 0.0)
        tile_dst_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 2048]), layout=(1, 128, 16), shape=(1, 128, 16)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(16, 64)),
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(16, 64)),
            )
            for kk in range(0, 640, 64):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 128, (cta.d0 * 640) + kk:(cta.d0 * 640) + kk + 64]),
                        layout=((128, 64), (5120, 1)),
                        shape=(128, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_ring_stages[(kk // 64) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(w[cta.d2 * 16:cta.d2 * 16 + 16, (cta.d0 * 640) + kk:(cta.d0 * 640) + kk + 64]),
                        layout=((16, 64), (5120, 1)),
                        shape=(16, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_ring_stages[(kk // 64) % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_2:
                    tile_2 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(kk // 64) % 2][0:0 + 16, 0:0 + 64]),
                        layout=Layout(((4, 16), (2, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(64, 16),
                    )
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
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_2[o_k:o_k + 16, o_n:o_n + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 16:o_k + 16 + 16, o_n:o_n + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 32:o_k + 32 + 16, o_n:o_n + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k + 48:o_k + 48 + 16, o_n:o_n + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_3),
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
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 16]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_2[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 16]),
                                        layout=(Layout((16, (2, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_3:
                tile_3 = T.tensor_view(
                    T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                    layout=((1, 2, 8, 2, 4, 2, 4, 2), (0, 1024, 1, 8, 16, 64, 128, 512)),
                    shape=(1, 128, 16),
                )
                with scope_3 as threads_5:
                    T.copy(tile_3, tile_dst_1)
                window = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, cta.d1 * 128:cta.d1 * 128 + 128, cta.d2 * 16:cta.d2 * 16 + 16]),
                    layout=((1, 128, 16), (12288, 96, 1)),
                    shape=(1, 128, 16),
                )
                T.copy(tile_dst_1, window)
