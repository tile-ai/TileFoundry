# analysis target=nvidia.h200_sxm module=QKV function=pieces topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r6.25MB/w20.00MB@logical,r3.87GB/w2.58GB@total,r30.00MB/w20.00MB@cta,r30.00MB/w16.02MB@thread;rmem:r84.01MB/w80.12MB@logical,r10.84GB/w10.83GB@total,r84.05MB/w84.00MB@cta,r391.50KB/w336.00KB@thread;smem:r30.00MB/w6.25MB@logical,r3.87GB/w3.87GB@total,r30.00MB/w30.00MB@cta,r5.16MB/w30.00MB@thread footprint=v38:484:16.00MB;w:256.00KB;x:1.00MB footprint-precision=exact peak=gmem:236.00MB;rmem:128.00KB;smem:144.00KB persistent=gmem:220.00MB
#   buffer=x holds=41.00MB time=bb space=none reuse=20.00MB fits=yes precision=exact
#   buffer=w holds=41.00MB time=bb space=none reuse=5.00MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def pieces(
    x: Tensor[(1, 8192, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], out: Tensor[(16384, 256), "f32"]
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
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 1, 0:0 + 8192, 0:0 + 5120]),
                layout=(8192, 5120),
                shape=(8192, 5120),
            )
            for bb in range(54, 56, 1):
                for mi in range(0, 16, 1):
                    for pc in range((((cta.d0 - (64 * bb)) - (4 * mi)) - (132 * ((((cta.d0 - (64 * bb)) - (4 * mi)) + 24) // 132))) + 24, 4, 132):
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
                        for kk in range(0, 20, 1):
                            with scope[2:, :32] as scope_1:
                                tile_1 = T.tensor_view(
                                    T.ptr_of(tile[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, ((pc * 20) + kk) * 64:((pc * 20) + kk) * 64 + 64]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                                    T.copy_async_tensor(tile_1, lhs_stages[kk % 3])
                                tile_2 = T.tensor_view(
                                    T.ptr_of(w[bb * 256:bb * 256 + 256, ((pc * 20) + kk) * 64:((pc * 20) + kk) * 64 + 64]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                                    T.copy_async_tensor(tile_2, rhs_stages[kk % 3])
                            with Mesh(
                                scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_2:
                                tile_3 = T.tensor_view(
                                    T.ptr_of(rhs_stages[kk % 3][0:0 + 256, 0:0 + 64]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m:o_m + 64, o_k:o_k + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
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
                                                    T.ptr_of(lhs_stages[kk % 3][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
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
                                acc_view_8 = T.tensor_view(
                                    T.ptr_of(acc[0:0 + 128, 0:0 + 256]),
                                    layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                    shape=(128, 256),
                                )
                                window = T.tensor_view(
                                    T.ptr_of(out[(((((bb - 54) * 16) + mi) * 4) + pc) * 128:(((((bb - 54) * 16) + mi) * 4) + pc) * 128 + 128, 0:0 + 256]),
                                    layout=(128, 256),
                                    shape=(128, 256),
                                )
                                T.copy(acc_view_8, window)
