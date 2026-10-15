# analysis target=nvidia.h200_sxm module=SCHED_GATE_UP_T128 function=gemm topology=cta wave=121/121
# selection requested=memory executed=memory
# memory traffic=gmem:r4.10MB/w4.32MB@logical,r495.82MB/w523.23MB@total,r4.10MB/w4.32MB@cta,r4.10MB/w4.32MB@thread;rmem:r6.54MB/w6.47MB@logical,r791.45MB/w782.72MB@total,r6.54MB/w6.47MB@cta,r28.08KB/w25.89KB@thread;smem:r5.35MB/w4.10MB@logical,r647.07MB/w495.82MB@total,r5.35MB/w4.10MB@cta,r796.00KB/w4.06MB@thread footprint=a:32.00KB;v45:64:36.00KB;w_gate:36.00KB;w_up:36.00KB footprint-precision=lower_bound peak=gmem:350.11MB;rmem:180.02KB;smem:208.00KB persistent=gmem:341.56MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 5120), "bf16"], w_gate: Tensor[(17424, 5120), "bf16"], w_up: Tensor[(17424, 5120), "bf16"], scratch: Tensor[(18432,), "bf16"], out: Tensor[(128, 17424), "bf16"]
):
    with Mesh((Topology("cta", 121),), layout=(1, 121), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        acc_g = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        acc_u = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        staged_dst = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(128, 144), shape=(128, 144)
        )
        tile_out = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        act = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        g = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        u = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "f32",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 144),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 18), (9216, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_dst_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 18432]), layout=(128, 144), shape=(128, 144)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_1:
                with scope_1 as threads:
                    T.fill(acc_g, 0.0)
                    T.fill(acc_u, 0.0)
            lhs_ring_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
            )
            gate_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
                T.tensor_view(36864, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
            )
            up_ring_stages = (
                T.tensor_view(73728, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
                T.tensor_view(110592, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
            )
            for k in range(0, 5120, 128):
                with scope[:1, :32] as scope_2:
                    tile = T.tensor_view(
                        T.ptr_of(a[cta.d0 * 128:cta.d0 * 128 + 128, k:k + 128]),
                        layout=((128, 128), (5120, 1)),
                        shape=(128, 128),
                    )
                    with Mesh(scope_2, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile, lhs_ring_stages[(k // 128) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(w_gate[cta.d1 * 144:cta.d1 * 144 + 144, k:k + 128]),
                        layout=((144, 128), (5120, 1)),
                        shape=(144, 128),
                    )
                    with Mesh(scope_2, layout=(32,), names=("d0",)) as threads_3:
                        T.copy_async_tensor(tile_1, gate_ring_stages[(k // 128) % 2])
                    tile_2 = T.tensor_view(
                        T.ptr_of(w_up[cta.d1 * 144:cta.d1 * 144 + 144, k:k + 128]),
                        layout=((144, 128), (5120, 1)),
                        shape=(144, 128),
                    )
                    with Mesh(scope_2, layout=(32,), names=("d0",)) as threads_4:
                        T.copy_async_tensor(tile_2, up_ring_stages[(k // 128) % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    tile_3 = T.tensor_view(
                        T.ptr_of(gate_ring_stages[(k // 128) % 2][0:0 + 144, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (18, 8)), ((9216, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 144),
                    )
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_5:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 144, 144):
                                for o_k in range(0, 128, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc_g[o_m:o_m + 64, o_n:o_n + 144]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_3[o_k:o_k + 16, o_n:o_n + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m:o_m + 64, o_n:o_n + 144]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k + 16:o_k + 16 + 16, o_n:o_n + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m:o_m + 64, o_n:o_n + 144]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k + 32:o_k + 32 + 16, o_n:o_n + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m:o_m + 64, o_n:o_n + 144]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k + 48:o_k + 48 + 16, o_n:o_n + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                    with Mesh(
                        scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_6:
                        for o_m_1 in range(64, 128, 64):
                            for o_n_1 in range(0, 144, 144):
                                for o_k_1 in range(0, 128, 64):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 144]),
                                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_6),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 144]),
                                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_6),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 144]),
                                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_6),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(acc_g[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 144]),
                                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_3[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_6),
                                    )
                    tile_4 = T.tensor_view(
                        T.ptr_of(up_ring_stages[(k // 128) % 2][0:0 + 144, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (18, 8)), ((9216, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 144),
                    )
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_7:
                        for o_m_2 in range(0, 64, 64):
                            for o_n_2 in range(0, 144, 144):
                                for o_k_2 in range(0, 128, 64):
                                    acc_view_8 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 144]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_8 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_8 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_8,
                                        lhs_view_8,
                                        rhs_view_8,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_7),
                                    )
                                    acc_view_9 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 144]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_9 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_9 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_9,
                                        lhs_view_9,
                                        rhs_view_9,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_7),
                                    )
                                    acc_view_10 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 144]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_10 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_10 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_10,
                                        lhs_view_10,
                                        rhs_view_10,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_7),
                                    )
                                    acc_view_11 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 144]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_11 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_11 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_11,
                                        lhs_view_11,
                                        rhs_view_11,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_7),
                                    )
                    with Mesh(
                        scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_8:
                        for o_m_3 in range(64, 128, 64):
                            for o_n_3 in range(0, 144, 144):
                                for o_k_3 in range(0, 128, 64):
                                    acc_view_12 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 144]),
                                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_12 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_12 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_12,
                                        lhs_view_12,
                                        rhs_view_12,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_8),
                                    )
                                    acc_view_13 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 144]),
                                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_13 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_13 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_13,
                                        lhs_view_13,
                                        rhs_view_13,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_8),
                                    )
                                    acc_view_14 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 144]),
                                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_14 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_14 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_14,
                                        lhs_view_14,
                                        rhs_view_14,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_8),
                                    )
                                    acc_view_15 = T.tensor_view(
                                        T.ptr_of(acc_u[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 144]),
                                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 18), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 144),
                                    )
                                    lhs_view_15 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[(k // 128) % 2][o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_15 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 144]),
                                        layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 144),
                                    )
                                    T.tiled_mma(
                                        acc_view_15,
                                        lhs_view_15,
                                        rhs_view_15,
                                        atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_8),
                                    )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                with scope_4 as threads_9:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc_g[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame, dst_frame, dtype="bf16")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(g[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_1, dst_frame_1, dtype="f32")
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(g[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.unary(src_frame_2, dst_frame_2, kind=UnaryKind.NEG)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_9.d0, 8 @ threads_9.d2, 2, 4 @ threads_9.d1, 2, 4 @ threads_9.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.unary(src_frame_3, dst_frame_3, kind=UnaryKind.EXP)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                with scope_4 as threads_13:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.binary(lhs_frame, scalar, dst_frame_4, kind=BinaryKind.ADD)
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(g[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.binary(lhs_frame_1, rhs_frame, dst_frame_5, kind=BinaryKind.DIV)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(act[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_4, dst_frame_6, dtype="bf16")
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(act[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_5, dst_frame_7, dtype="f32")
                    src_frame_6 = T.tensor_view(
                        T.ptr_of(acc_u[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_6, dst_frame_8, dtype="bf16")
                    src_frame_7 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(u[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_7, dst_frame_9, dtype="f32")
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(u[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(value[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.binary(lhs_frame_2, rhs_frame_1, dst_frame_10, kind=BinaryKind.MUL)
                    src_frame_8 = T.tensor_view(
                        T.ptr_of(value[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(tile_out[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.cast(src_frame_8, dst_frame_11, dtype="bf16")
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(tile_out[0:0 + 128, 0:0 + 144]),
                        layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 18), (9216, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 144),
                    )
                    T.copy(src_frame_9, staged_dst)
            with scope[:1, :32] as scope_5:
                with Mesh(scope_5, layout=(32,), names=("d0",)) as threads_22:
                    T.copy_async_tensor(staged_dst, tile_dst_1)
                window = T.tensor_view(
                    T.ptr_of(out[cta.d0 * 128:cta.d0 * 128 + 128, cta.d1 * 144:cta.d1 * 144 + 144]),
                    layout=((128, 144), (17424, 1)),
                    shape=(128, 144),
                )
                T.copy(tile_dst_1, window)
