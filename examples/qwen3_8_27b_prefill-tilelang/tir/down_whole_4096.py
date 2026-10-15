# analysis target=nvidia.h200_sxm module=Residual function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r196.38MB/w45.12MB@logical,r68.05GB/w5.82GB@total,r527.88MB/w45.12MB@cta,r527.88MB/w45.12MB@thread;rmem:r1.37GB/w1.37GB@logical,r181.39GB/w181.35GB@total,r1.37GB/w1.37GB@cta,r5.84MB/w5.50MB@thread;smem:r527.88MB/w196.38MB@logical,r68.05GB/w68.05GB@total,r527.88MB/w527.88MB@cta,r92.42MB/w525.32MB@thread footprint=hidden:8.25MB;v34:149:128.00KB;v35:150:8.25MB;v64:170:0;v65:171:0;w:288.00KB;x:256.00KB footprint-precision=exact peak=gmem:386.06MB;rmem:256.00KB;smem:208.00KB persistent=gmem:346.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 4096, 17408), "bf16"], hidden: Tensor[(1, 4096, 5120), "bf16"], w: Tensor[(5120, 17408), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(4096, 5120), "bf16"]
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
        y = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
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
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        acc_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        y_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 4, 2, 8, 32, 4, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 1, 0:0 + 4096, 0:0 + 17408]),
                layout=(4096, 17408),
                shape=(4096, 17408),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(hidden[0:0 + 1, 0:0 + 4096, 0:0 + 5120]),
                layout=(4096, 5120),
                shape=(4096, 5120),
            )
            for g in range(0, 2, 1):
                for bn in range(0, 20, 1):
                    res = T.tensor_view(
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
                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    for mi in range(((cta.d0 - (56 * g)) - (16 * bn)) - (132 * (((cta.d0 - (56 * g)) - (16 * bn)) // 132)), 16, 132):
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
                        for k in range(0, 17408, 64):
                            with scope[2:, :32] as scope_1:
                                tile_2 = T.tensor_view(
                                    T.ptr_of(tile[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 128, k:k + 64]),
                                    layout=((128, 64), (17408, 1)),
                                    shape=(128, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                                    T.copy_async_tensor(tile_2, lhs_stages[(k // 64) % 3])
                                tile_3 = T.tensor_view(
                                    T.ptr_of(w[bn * 256:bn * 256 + 256, k:k + 64]),
                                    layout=((256, 64), (17408, 1)),
                                    shape=(256, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                                    T.copy_async_tensor(tile_3, rhs_stages[(k // 64) % 3])
                            with Mesh(
                                scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_2:
                                tile_4 = T.tensor_view(
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
                                                    T.ptr_of(tile_4[o_k:o_k + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(tile_4[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(tile_4[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(tile_4[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
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
                                                    T.ptr_of(tile_4[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(tile_4[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(tile_4[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
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
                                                    T.ptr_of(tile_4[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                    shape=(16, 256),
                                                )
                                                T.tiled_mma(
                                                    acc_view_7,
                                                    lhs_view_7,
                                                    rhs_view_7,
                                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                                )
                        with scope[:1, :32] as scope_3:
                            tile_5 = T.tensor_view(
                                T.ptr_of(tile_1[((g * 16) + mi) * 128:((g * 16) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                                layout=((128, 256), (5120, 1)),
                                shape=(128, 256),
                            )
                            with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_5:
                                T.copy_async_tensor(tile_5, res)
                        with Mesh(
                            scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_4:
                            with scope_4 as threads_6:
                                dst_frame = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.copy(res, dst_frame)
                                src_frame = T.tensor_view(
                                    T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                dst_frame_1 = T.tensor_view(
                                    T.ptr_of(value_1[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.cast(src_frame, dst_frame_1, dtype="bf16")
                                lhs_frame = T.tensor_view(
                                    T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                rhs_frame = T.tensor_view(
                                    T.ptr_of(value_1[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                dst_frame_2 = T.tensor_view(
                                    T.ptr_of(y[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.binary(lhs_frame, rhs_frame, dst_frame_2, kind=BinaryKind.ADD)
                                src_frame_1 = T.tensor_view(
                                    T.ptr_of(y[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_6.d0, 4 @ threads_6.d1, 2, 8 @ threads_6.d2, 32, 4 @ threads_6.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                T.copy(src_frame_1, staged)
                        with scope[:1, :32] as scope_5:
                            with Mesh(scope_5, layout=(32,), names=("d0",)) as threads_10:
                                window = T.tensor_view(
                                    T.ptr_of(out[(2048 * g) + (128 * mi):(2048 * g) + (128 * mi) + 128, 256 * bn:256 * bn + 256]),
                                    layout=((128, 256), (5120, 1)),
                                    shape=(128, 256),
                                )
                                T.copy_async_tensor(staged, window)
            for bn in range(0, 0, 1):
                res_1 = T.tensor_view(
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
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                tile_10_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
                )
                for mi in range(((cta.d0 - (16 * bn)) - (132 * (((cta.d0 - (16 * bn)) - 56) // 132))) - 56, 16, 132):
                    with Mesh(
                        scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as threads_11:
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
                    for k in range(0, 17408, 64):
                        with scope[2:, :32] as scope_6:
                            tile_6 = T.tensor_view(
                                T.ptr_of(tile[((1 * 16) + mi) * 128:((1 * 16) + mi) * 128 + 128, k:k + 64]),
                                layout=((128, 64), (17408, 1)),
                                shape=(128, 64),
                            )
                            with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_12:
                                T.copy_async_tensor(tile_6, lhs_1_stages[(k // 64) % 3])
                            tile_7 = T.tensor_view(
                                T.ptr_of(w[bn * 256:bn * 256 + 256, k:k + 64]),
                                layout=((256, 64), (17408, 1)),
                                shape=(256, 64),
                            )
                            with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_13:
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
                            ) as threads_14:
                                for o_m_2 in range(0, 64, 64):
                                    for o_n_2 in range(0, 256, 256):
                                        for o_k_2 in range(0, 64, 64):
                                            acc_view_8 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((4 @ threads_14.d0, 2, 8 @ threads_14.d1, 32, 4 @ threads_14.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                            )
                                            acc_view_9 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((4 @ threads_14.d0, 2, 8 @ threads_14.d1, 32, 4 @ threads_14.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                            )
                                            acc_view_10 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((4 @ threads_14.d0, 2, 8 @ threads_14.d1, 32, 4 @ threads_14.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                            )
                                            acc_view_11 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                                layout=((4 @ threads_14.d0, 2, 8 @ threads_14.d1, 32, 4 @ threads_14.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                            )
                            with Mesh(
                                scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                            ) as threads_15:
                                for o_m_3 in range(64, 128, 64):
                                    for o_n_3 in range(0, 256, 256):
                                        for o_k_3 in range(0, 64, 64):
                                            acc_view_12 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((4 @ threads_15.d0, 2, 8 @ threads_15.d1, 32, 4 @ threads_15.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                            )
                                            acc_view_13 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((4 @ threads_15.d0, 2, 8 @ threads_15.d1, 32, 4 @ threads_15.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                            )
                                            acc_view_14 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((4 @ threads_15.d0, 2, 8 @ threads_15.d1, 32, 4 @ threads_15.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                            )
                                            acc_view_15 = T.tensor_view(
                                                T.ptr_of(acc_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                                layout=((4 @ threads_15.d0, 2, 8 @ threads_15.d1, 32, 4 @ threads_15.d2, 2,), (16, 8, 1, 512, 128, 64)),
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
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                            )
                    with scope[:1, :32] as scope_8:
                        tile_9 = T.tensor_view(
                            T.ptr_of(tile_1[((1 * 16) + mi) * 128:((1 * 16) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        with Mesh(scope_8, layout=(32,), names=("d0",)) as threads_16:
                            T.copy_async_tensor(tile_9, res_1)
                    with Mesh(
                        scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_9:
                        with scope_9 as threads_17:
                            dst_frame_3 = T.tensor_view(
                                T.ptr_of(value_2[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.copy(res_1, dst_frame_3)
                            src_frame_2 = T.tensor_view(
                                T.ptr_of(acc_1[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            dst_frame_4 = T.tensor_view(
                                T.ptr_of(value_3[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(src_frame_2, dst_frame_4, dtype="bf16")
                            lhs_frame_1 = T.tensor_view(
                                T.ptr_of(value_2[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            rhs_frame_1 = T.tensor_view(
                                T.ptr_of(value_3[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            dst_frame_5 = T.tensor_view(
                                T.ptr_of(y_1[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.binary(lhs_frame_1, rhs_frame_1, dst_frame_5, kind=BinaryKind.ADD)
                            src_frame_3 = T.tensor_view(
                                T.ptr_of(y_1[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_17.d0, 4 @ threads_17.d1, 2, 8 @ threads_17.d2, 32, 4 @ threads_17.d3, 2,), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.copy(src_frame_3, staged_1)
                    with scope[:1, :32] as scope_10:
                        with Mesh(scope_10, layout=(32,), names=("d0",)) as threads_21:
                            T.copy_async_tensor(staged_1, tile_10_1)
                        window_1 = T.tensor_view(
                            T.ptr_of(out[((1 * 16) + mi) * 128:((1 * 16) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                            layout=((128, 256), (5120, 1)),
                            shape=(128, 256),
                        )
                        T.copy(tile_10_1, window_1)
