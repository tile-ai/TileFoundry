# analysis target=nvidia.h200_sxm module=GateUp function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r354.78MB/w153.06MB@logical,r133.07GB/w19.73GB@total,r1.01GB/w153.06MB@cta,r1.01GB/w153.06MB@thread;rmem:r2.85GB/w2.80GB@logical,r376.20GB/w373.91GB@total,r2.85GB/w2.83GB@cta,r12.10MB/w11.33MB@thread;smem:r1.02GB/w371.84MB@logical,r135.27GB/w135.27GB@total,r1.02GB/w1.02GB@cta,r184.55MB/w1023.85MB@thread footprint=v41:289:64.00KB;v43:290:4.12MB;v80:312:0;v82:313:0;w:288.00KB;x:256.00KB footprint-precision=exact peak=gmem:516.03MB;rmem:192.00KB;smem:208.00KB persistent=gmem:380.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 4096, 5120), "bf16"], w: Tensor[(34816, 5120), "bf16"], scratch: Tensor[(16384,), "bf16"], out: Tensor[(4096, 17408), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        act = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        value_1 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        g32 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        value_4 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_5 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_6 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        u = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        acc_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        act_1 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        value_8 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        value_9 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        g32_1 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"]
        )
        value_11 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_12 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        value_13 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "f32", (16, 8, 16, 8), "rmem"])
        u_1 = T.alloc_tensor(tensor_type=Tensor[(128, 128), "bf16", (16, 8, 16, 8), "rmem"])
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 1, 0:0 + 4096, 0:0 + 5120]),
                layout=(4096, 5120),
                shape=(4096, 5120),
            )
            for g in range(0, 2, 1):
                for bn in range(0, 136, 1):
                    cs = T.tensor_view(
                        147456,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    staged = T.tensor_view(
                        147456,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    for mi in range(((cta.d0 - (64 * g)) - (16 * bn)) - (132 * (((cta.d0 - (64 * g)) - (16 * bn)) // 132)), 16, 132):
                        with Mesh(
                            scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as threads:
                            T.fill(acc, 0.0)
                        lhs_stages = (
                            T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                            T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                            T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                        )
                        rhs_stages = (
                            T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                            T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                            T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                        )
                        for k in range(0, 5120, 64):
                            with scope[2:, :32] as scope_1:
                                tile_1 = T.tensor_view(
                                    T.ptr_of(tile[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 128, k:k + 64]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                                    T.copy_async_tensor(tile_1, lhs_stages[(k // 64) % 3])
                                tile_2 = T.tensor_view(
                                    T.ptr_of(w[bn * 256:bn * 256 + 256, k:k + 64]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                                    T.copy_async_tensor(tile_2, rhs_stages[(k // 64) % 3])
                            with Mesh(
                                scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_2:
                                tile_3 = T.tensor_view(
                                    T.ptr_of(rhs_stages[(k // 64) % 3][0:0 + 256, 0:0 + 64]),
                                    layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                with Mesh(
                                    scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")
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
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k:o_k + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_1 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_2 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_3 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
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
                                    scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
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
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_4 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_5 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_6 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(lhs_stages[(k // 64) % 3][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(64, 16),
                                                )
                                                rhs_view_7 = T.tensor_view(
                                                    T.ptr_of(tile_3[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
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
                            scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_3:
                            with scope_3 as threads_5:
                                src_frame = T.tensor_view(
                                    T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                    shape=(128, 256),
                                )
                                dst_frame = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                    shape=(128, 256),
                                )
                                T.cast(src_frame, dst_frame, dtype="bf16")
                                src_frame_1 = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                    shape=(128, 256),
                                )
                                T.copy(src_frame_1, cs)
                        with Mesh(scope[:2], layout=(16, 16), names=("d0", "d1")) as scope_4:
                            tile_4 = T.tensor_view(
                                T.ptr_of(cs[0:0 + 128, 0:0 + 128]),
                                layout=Layout((128, 128), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            with scope_4 as threads_7:
                                dst_frame_1 = T.tensor_view(
                                    T.ptr_of(value_3[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 8),
                                    shape=(128, 128),
                                )
                                T.copy(tile_4, dst_frame_1)
                                src_frame_2 = T.tensor_view(
                                    T.ptr_of(value_3[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_2 = T.tensor_view(
                                    T.ptr_of(g32[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 8),
                                    shape=(128, 128),
                                )
                                T.cast(src_frame_2, dst_frame_2, dtype="f32")
                            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                            T.fill(scalar, 1.0)
                            with scope_4 as threads_9:
                                src_frame_3 = T.tensor_view(
                                    T.ptr_of(g32[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_3 = T.tensor_view(
                                    T.ptr_of(value_6[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                T.unary(src_frame_3, dst_frame_3, kind=UnaryKind.NEG)
                                src_frame_4 = T.tensor_view(
                                    T.ptr_of(value_6[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_4 = T.tensor_view(
                                    T.ptr_of(value_5[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                T.unary(src_frame_4, dst_frame_4, kind=UnaryKind.EXP)
                                rhs_frame = T.tensor_view(
                                    T.ptr_of(value_5[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_5 = T.tensor_view(
                                    T.ptr_of(value_4[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                T.binary(scalar, rhs_frame, dst_frame_5, kind=BinaryKind.ADD)
                                lhs_frame = T.tensor_view(
                                    T.ptr_of(g32[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                rhs_frame_1 = T.tensor_view(
                                    T.ptr_of(value_4[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_6 = T.tensor_view(
                                    T.ptr_of(value_2[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                T.binary(
                                    lhs_frame,
                                    rhs_frame_1,
                                    dst_frame_6,
                                    kind=BinaryKind.DIV,
                                )
                                src_frame_5 = T.tensor_view(
                                    T.ptr_of(value_2[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_7 = T.tensor_view(
                                    T.ptr_of(value_1[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_9.d0, 8, 16 @ threads_9.d1, 8),
                                    shape=(128, 128),
                                )
                                T.cast(src_frame_5, dst_frame_7, dtype="bf16")
                            tile_5 = T.tensor_view(
                                T.ptr_of(cs[0:0 + 128, 128:128 + 128]),
                                layout=Layout((128, 128), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 128),
                            )
                            with scope_4 as threads_14:
                                dst_frame_8 = T.tensor_view(
                                    T.ptr_of(u[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_14.d0, 8, 16 @ threads_14.d1, 8),
                                    shape=(128, 128),
                                )
                                T.copy(tile_5, dst_frame_8)
                                lhs_frame_1 = T.tensor_view(
                                    T.ptr_of(value_1[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_14.d0, 8, 16 @ threads_14.d1, 8),
                                    shape=(128, 128),
                                )
                                rhs_frame_2 = T.tensor_view(
                                    T.ptr_of(u[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_14.d0, 8, 16 @ threads_14.d1, 8),
                                    shape=(128, 128),
                                )
                                dst_frame_9 = T.tensor_view(
                                    T.ptr_of(act[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_14.d0, 8, 16 @ threads_14.d1, 8),
                                    shape=(128, 128),
                                )
                                T.binary(
                                    lhs_frame_1,
                                    rhs_frame_2,
                                    dst_frame_9,
                                    kind=BinaryKind.MUL,
                                )
                                src_frame_6 = T.tensor_view(
                                    T.ptr_of(act[0:0 + 128, 0:0 + 128]),
                                    layout=(16 @ threads_14.d0, 8, 16 @ threads_14.d1, 8),
                                    shape=(128, 128),
                                )
                                T.copy(src_frame_6, staged)
                        with scope[:1, :32] as scope_5:
                            with Mesh(scope_5, layout=(32,), names=("d0",)) as threads_17:
                                window = T.tensor_view(
                                    T.ptr_of(out[(2048 * g) + (128 * mi):(2048 * g) + (128 * mi) + 128, 128 * bn:128 * bn + 128]),
                                    layout=((128, 128), (17408, 1)),
                                    shape=(128, 128),
                                )
                                T.copy_async_tensor(staged, window)
            for bn in range(0, 0, 1):
                cs_1 = T.tensor_view(
                    147456,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                staged_1 = T.tensor_view(
                    147456,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 128),
                )
                tile_11_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 16384]), layout=(128, 128), shape=(128, 128)
                )
                for mi in range(((cta.d0 - (16 * bn)) - (132 * (((cta.d0 - (16 * bn)) - 64) // 132))) - 64, 16, 132):
                    with Mesh(
                        scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as threads_18:
                        T.fill(acc_1, 0.0)
                    lhs_1_stages = (
                        T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                        T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                        T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                    )
                    rhs_1_stages = (
                        T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                        T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                        T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                    )
                    for k in range(0, 5120, 64):
                        with scope[2:, :32] as scope_6:
                            tile_6 = T.tensor_view(
                                T.ptr_of(tile[((1 * 16) + mi) * 128:((1 * 16) + mi) * 128 + 128, k:k + 64]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_19:
                                T.copy_async_tensor(tile_6, lhs_1_stages[(k // 64) % 3])
                            tile_7 = T.tensor_view(
                                T.ptr_of(w[bn * 256:bn * 256 + 256, k:k + 64]),
                                layout=((256, 64), (5120, 1)),
                                shape=(256, 64),
                            )
                            with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_20:
                                T.copy_async_tensor(tile_7, rhs_1_stages[(k // 64) % 3])
                        with Mesh(
                            scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_7:
                            tile_8 = T.tensor_view(
                                T.ptr_of(rhs_1_stages[(k // 64) % 3][0:0 + 256, 0:0 + 64]),
                                layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            with Mesh(
                                scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")
                            ) as threads_21:
                                for o_m_2 in range(0, 64, 64):
                                    for o_n_2 in range(0, 256, 256):
                                        for o_k_2 in range(0, 64, 64):
                                            acc_view_8 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_8 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_8 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_8,
                                                lhs_view_8,
                                                rhs_view_8,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_21),
                                            )
                                            acc_view_9 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_9 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_9 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_9,
                                                lhs_view_9,
                                                rhs_view_9,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_21),
                                            )
                                            acc_view_10 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_10 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_10 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_10,
                                                lhs_view_10,
                                                rhs_view_10,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_21),
                                            )
                                            acc_view_11 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((8 @ threads_21.d1, 2, 4 @ threads_21.d0, 2, 4 @ threads_21.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_11 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_11 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_11,
                                                lhs_view_11,
                                                rhs_view_11,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_21),
                                            )
                            with Mesh(
                                scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                            ) as threads_22:
                                for o_m_3 in range(64, 128, 64):
                                    for o_n_3 in range(0, 256, 256):
                                        for o_k_3 in range(0, 64, 64):
                                            acc_view_12 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_12 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_12 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_12,
                                                lhs_view_12,
                                                rhs_view_12,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_22),
                                            )
                                            acc_view_13 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_13 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_13 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_13,
                                                lhs_view_13,
                                                rhs_view_13,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_22),
                                            )
                                            acc_view_14 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_14 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_14 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_14,
                                                lhs_view_14,
                                                rhs_view_14,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_22),
                                            )
                                            acc_view_15 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_15 = T.tensor_view(
                                                T.ptr_of(lhs_1_stages[(k // 64) % 3][o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(64, 16),
                                            )
                                            rhs_view_15 = T.tensor_view(
                                                T.ptr_of(tile_8[o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_15,
                                                lhs_view_15,
                                                rhs_view_15,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_22),
                                            )
                    with Mesh(
                        scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_8:
                        with scope_8 as threads_23:
                            src_frame_7 = T.tensor_view(
                                T.ptr_of(acc_1[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_23.d0, 8 @ threads_23.d2, 2, 4 @ threads_23.d1, 2, 4 @ threads_23.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            dst_frame_10 = T.tensor_view(
                                T.ptr_of(value_7[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_23.d0, 8 @ threads_23.d2, 2, 4 @ threads_23.d1, 2, 4 @ threads_23.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.cast(src_frame_7, dst_frame_10, dtype="bf16")
                            src_frame_8 = T.tensor_view(
                                T.ptr_of(value_7[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_23.d0, 8 @ threads_23.d2, 2, 4 @ threads_23.d1, 2, 4 @ threads_23.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.copy(src_frame_8, cs_1)
                    with Mesh(scope[:2], layout=(16, 16), names=("d0", "d1")) as scope_9:
                        tile_9 = T.tensor_view(
                            T.ptr_of(cs_1[0:0 + 128, 0:0 + 128]),
                            layout=Layout((128, 128), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        with scope_9 as threads_25:
                            dst_frame_11 = T.tensor_view(
                                T.ptr_of(value_10[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_25.d0, 8, 16 @ threads_25.d1, 8),
                                shape=(128, 128),
                            )
                            T.copy(tile_9, dst_frame_11)
                            src_frame_9 = T.tensor_view(
                                T.ptr_of(value_10[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_25.d0, 8, 16 @ threads_25.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_12 = T.tensor_view(
                                T.ptr_of(g32_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_25.d0, 8, 16 @ threads_25.d1, 8),
                                shape=(128, 128),
                            )
                            T.cast(src_frame_9, dst_frame_12, dtype="f32")
                        scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_1, 1.0)
                        with scope_9 as threads_27:
                            src_frame_10 = T.tensor_view(
                                T.ptr_of(g32_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_13 = T.tensor_view(
                                T.ptr_of(value_13[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            T.unary(src_frame_10, dst_frame_13, kind=UnaryKind.NEG)
                            src_frame_11 = T.tensor_view(
                                T.ptr_of(value_13[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_14 = T.tensor_view(
                                T.ptr_of(value_12[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            T.unary(src_frame_11, dst_frame_14, kind=UnaryKind.EXP)
                            rhs_frame_3 = T.tensor_view(
                                T.ptr_of(value_12[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_15 = T.tensor_view(
                                T.ptr_of(value_11[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            T.binary(scalar_1, rhs_frame_3, dst_frame_15, kind=BinaryKind.ADD)
                            lhs_frame_2 = T.tensor_view(
                                T.ptr_of(g32_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            rhs_frame_4 = T.tensor_view(
                                T.ptr_of(value_11[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_16 = T.tensor_view(
                                T.ptr_of(value_9[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            T.binary(
                                lhs_frame_2,
                                rhs_frame_4,
                                dst_frame_16,
                                kind=BinaryKind.DIV,
                            )
                            src_frame_12 = T.tensor_view(
                                T.ptr_of(value_9[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_17 = T.tensor_view(
                                T.ptr_of(value_8[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_27.d0, 8, 16 @ threads_27.d1, 8),
                                shape=(128, 128),
                            )
                            T.cast(src_frame_12, dst_frame_17, dtype="bf16")
                        tile_10 = T.tensor_view(
                            T.ptr_of(cs_1[0:0 + 128, 128:128 + 128]),
                            layout=Layout((128, 128), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 128),
                        )
                        with scope_9 as threads_32:
                            dst_frame_18 = T.tensor_view(
                                T.ptr_of(u_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                                shape=(128, 128),
                            )
                            T.copy(tile_10, dst_frame_18)
                            lhs_frame_3 = T.tensor_view(
                                T.ptr_of(value_8[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                                shape=(128, 128),
                            )
                            rhs_frame_5 = T.tensor_view(
                                T.ptr_of(u_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                                shape=(128, 128),
                            )
                            dst_frame_19 = T.tensor_view(
                                T.ptr_of(act_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                                shape=(128, 128),
                            )
                            T.binary(
                                lhs_frame_3,
                                rhs_frame_5,
                                dst_frame_19,
                                kind=BinaryKind.MUL,
                            )
                            src_frame_13 = T.tensor_view(
                                T.ptr_of(act_1[0:0 + 128, 0:0 + 128]),
                                layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                                shape=(128, 128),
                            )
                            T.copy(src_frame_13, staged_1)
                    with scope[:1, :32] as scope_10:
                        with Mesh(scope_10, layout=(32,), names=("d0",)) as threads_35:
                            T.copy_async_tensor(staged_1, tile_11_1)
                        window_1 = T.tensor_view(
                            T.ptr_of(out[((1 * 16) + mi) * 128:((1 * 16) + mi) * 128 + 128, bn * 128:bn * 128 + 128]),
                            layout=((128, 128), (17408, 1)),
                            shape=(128, 128),
                        )
                        T.copy(tile_11_1, window_1)
