# analysis target=nvidia.h200_sxm module=InProj function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r167.81MB/w40.62MB@logical,r31.94GB/w5.24GB@total,r247.81MB/w40.62MB@cta,r247.81MB/w40.62MB@thread;rmem:r662.27MB/w654.19MB@logical,r85.38GB/w85.36GB@total,r662.35MB/w662.19MB@cta,r2.75MB/w2.59MB@thread;smem:r247.81MB/w167.81MB@logical,r31.94GB/w31.94GB@total,r247.81MB/w247.81MB@cta,r45.96MB/w243.77MB@thread footprint=v29:398:128.00KB;v30:399:8.25MB;w:544.00KB;x:128.00KB footprint-precision=exact peak=gmem:205.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:172.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 5120), "bf16"], w: Tensor[(16640, 5120), "bf16"], out: Tensor[(1024, 16640), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 1, 0:0 + 1024, 0:0 + 5120]),
                layout=(1024, 5120),
                shape=(1024, 5120),
            )
            for g in range(0, 1, 1):
                for bn in range(0, 65, 1):
                    staged = T.tensor_view(
                        147456,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    for mi in range(((cta.d0 + (8 * g)) - (8 * bn)) - (132 * (((cta.d0 + (8 * g)) - (8 * bn)) // 132)), 8, 132):
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
                                    T.ptr_of(tile[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 128, k:k + 64]),
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
                                                    layout=((4 @ threads_3.d0, 2, 8 @ threads_3.d1, 32, 4 @ threads_3.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_3.d0, 2, 8 @ threads_3.d1, 32, 4 @ threads_3.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_3.d0, 2, 8 @ threads_3.d1, 32, 4 @ threads_3.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_3.d0, 2, 8 @ threads_3.d1, 32, 4 @ threads_3.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_4.d0, 2, 8 @ threads_4.d1, 32, 4 @ threads_4.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_4.d0, 2, 8 @ threads_4.d1, 32, 4 @ threads_4.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_4.d0, 2, 8 @ threads_4.d1, 32, 4 @ threads_4.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                    layout=((4 @ threads_4.d0, 2, 8 @ threads_4.d1, 32, 4 @ threads_4.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                    layout=((2 @ threads_5.d0, 4 @ threads_5.d1, 2, 8 @ threads_5.d2, 32, 4 @ threads_5.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                dst_frame = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 4 @ threads_5.d1, 2, 8 @ threads_5.d2, 32, 4 @ threads_5.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.cast(src_frame, dst_frame, dtype="bf16")
                                src_frame_1 = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 4 @ threads_5.d1, 2, 8 @ threads_5.d2, 32, 4 @ threads_5.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.copy(src_frame_1, staged)
                        with scope[:1, :32] as scope_4:
                            with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_7:
                                window = T.tensor_view(
                                    T.ptr_of(out[128 * mi:128 * mi + 128, 256 * bn:256 * bn + 256]),
                                    layout=((128, 256), (16640, 1)),
                                    shape=(128, 256),
                                )
                                T.copy_async_tensor(staged, window)
