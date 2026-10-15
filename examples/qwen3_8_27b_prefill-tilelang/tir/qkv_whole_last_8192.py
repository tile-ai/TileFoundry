# analysis target=nvidia.h200_sxm module=QKV function=whole_last topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r139.69MB/w6.75MB@logical,r26.98GB/w891.00MB@total,r209.30MB/w6.75MB@cta,r205.89MB/w6.75MB@thread;rmem:r552.98MB/w545.73MB@logical,r90.36GB/w86.42GB@total,r700.94MB/w670.41MB@cta,r2.92MB/w2.65MB@thread;smem:r206.03MB/w139.78MB@logical,r27.63GB/w27.63GB@total,r214.31MB/w214.31MB@cta,r38.21MB/w202.55MB@thread footprint=cos:256.00KB;gamma_k:512B;gamma_q:512B;sin:256.00KB;v164:712:128.00KB;v165:713:8.25MB;w:288.00KB;x:256.00KB footprint-precision=exact peak=gmem:670.00MB;rmem:256.50KB;smem:208.00KB persistent=gmem:446.00MB
#   buffer=cos holds=51.38MB time=bn space=none reuse=13.25MB fits=no precision=exact
#   buffer=sin holds=51.38MB time=bn space=none reuse=13.25MB fits=no precision=exact
#   buffer=gamma_q holds=51.38MB time=bn space=cta.persistent reuse=3.48MB fits=no precision=exact
#   buffer=gamma_k holds=51.38MB time=bn space=cta.persistent reuse=3.48MB fits=no precision=exact
#   error="rmem placement peak 256.50KB exceeds capacity 256.00KB"
#   error="l2 reuse window bn holds 51.38MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole_last(
    out: Tensor[(8192, 14336), "bf16"], x: Tensor[(1, 8192, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], gamma_q: Tensor[(256,), "bf16"], gamma_k: Tensor[(256,), "bf16"], cos: Tensor[(8192, 64), "bf16"], sin: Tensor[(8192, 64), "bf16"], scratch: Tensor[(117440512,), "bf16"], out_1: Tensor[(8192, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        out_2_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 117440512]), layout=(8192, 14336), shape=(8192, 14336)
        )
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
        nrm = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "bf16", (16, 8, 16, 16), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        x32 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "bf16", (16, 8, 16, 16), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"]
        )
        ms = T.alloc_tensor(tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"])
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[
                (16, 8, 1, 1), "f32", ((16, 8, 1, 1, 1), (8, 1, 0, 0, 0)), "rmem"
            ]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", ((16, 8, 16, 16), (0, 16, 0, 1)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        gw = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 16), "f32", (1, 1, 16, 16), "rmem"])
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 16), "f32", (1, 1, 16, 16), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 16), "bf16", (1, 1, 16, 16), "rmem"]
        )
        rlo = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"])
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        lo = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        cl = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        hi = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        sl = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        rhi = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"])
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        ch = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        sh = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        nrm_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "bf16", (16, 8, 16, 16), "rmem"]
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        x32_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        value_33 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "bf16", (16, 8, 16, 16), "rmem"]
        )
        value_34 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"]
        )
        value_35 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"]
        )
        ms_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 1, 1), "f32", (16, 8, 1, 1), "rmem"])
        value_36 = T.alloc_tensor(
            tensor_type=Tensor[
                (16, 8, 1, 1), "f32", ((16, 8, 1, 1, 1), (8, 1, 0, 0, 0)), "rmem"
            ]
        )
        sq_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", ((16, 8, 16, 16), (0, 16, 0, 1)), "rmem"]
        )
        value_37 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 16), "f32", (16, 8, 16, 16), "rmem"]
        )
        gw_1 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 16), "f32", (1, 1, 16, 16), "rmem"])
        value_38 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 16), "f32", (1, 1, 16, 16), "rmem"]
        )
        value_39 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 16), "bf16", (1, 1, 16, 16), "rmem"]
        )
        rlo_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_40 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_41 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_42 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_43 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        lo_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_44 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        cl_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_45 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_46 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_47 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_48 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_49 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        hi_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_50 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        sl_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_51 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        rhi_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_52 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_53 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_54 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_55 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        ch_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_56 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_57 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        value_58 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        value_59 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"]
        )
        sh_1 = T.alloc_tensor(tensor_type=Tensor[(16, 8, 16, 2), "f32", (16, 8, 16, 2), "rmem"])
        value_60 = T.alloc_tensor(
            tensor_type=Tensor[(16, 8, 16, 2), "bf16", (16, 8, 16, 2), "rmem"]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(x[0:0 + 1, 0:0 + 8192, 0:0 + 5120]),
                layout=(8192, 5120),
                shape=(8192, 5120),
            )
            T.copy(out, out_2_1)
            for bn in range(0, 54, 1):
                cs = T.tensor_view(
                    147456,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                for mi in range(((cta.d0 - (16 * bn)) - (132 * (((cta.d0 - (16 * bn)) - 48) // 132))) - 48, 16, 132):
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
                                T.ptr_of(tile[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, k:k + 64]),
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
                            T.copy(src_frame_1, cs)
                    for _q in range(bn - (56 * (bn // 56)), 48, 1048576):
                        cs_1 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        for _h in range(bn - (2 * (bn // 2)), 1, 2):
                            with Mesh(
                                scope[:2], layout=(16, 16), names=("d0", "d1")
                            ) as scope_4:
                                tile_4 = T.tensor_view(
                                    T.ptr_of(cs[0:0 + 128, 0:0 + 256]),
                                    layout=(16, 8, 16, 16) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 16),
                                )
                                with scope_4 as threads_7:
                                    dst_frame_1 = T.tensor_view(
                                        T.ptr_of(value_3[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(tile_4, dst_frame_1)
                                    src_frame_2 = T.tensor_view(
                                        T.ptr_of(value_3[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_2 = T.tensor_view(
                                        T.ptr_of(x32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_frame_2, dst_frame_2, dtype="f32")
                                    lhs_frame = T.tensor_view(
                                        T.ptr_of(x32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame = T.tensor_view(
                                        T.ptr_of(x32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_3 = T.tensor_view(
                                        T.ptr_of(value_7[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame,
                                        rhs_frame,
                                        dst_frame_3,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_3 = T.tensor_view(
                                        T.ptr_of(value_7[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_4 = T.tensor_view(
                                        T.ptr_of(sq[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(src_frame_3, dst_frame_4)
                                    src_frame_4 = T.tensor_view(
                                        T.ptr_of(sq[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ threads_7.d0, 8, 16 @ threads_7.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_5 = T.tensor_view(
                                        T.ptr_of(value_6[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=((16 @ threads_7.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.reduce(
                                        src_frame_4,
                                        dst_frame_5,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    src_frame_5 = T.tensor_view(
                                        T.ptr_of(value_6[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=((16 @ threads_7.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_6 = T.tensor_view(
                                        T.ptr_of(ms[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_7.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(src_frame_5, dst_frame_6)
                                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar, 1e-06)
                                with scope_4 as threads_13:
                                    lhs_frame_1 = T.tensor_view(
                                        T.ptr_of(ms[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_13.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_7 = T.tensor_view(
                                        T.ptr_of(value_5[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_13.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(
                                        lhs_frame_1,
                                        scalar,
                                        dst_frame_7,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_6 = T.tensor_view(
                                        T.ptr_of(value_5[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_13.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_8 = T.tensor_view(
                                        T.ptr_of(value_4[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_13.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.unary(src_frame_6, dst_frame_8, kind=UnaryKind.RSQRT)
                                    lhs_frame_2 = T.tensor_view(
                                        T.ptr_of(x32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_13.d0, 8, 16 @ threads_13.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame_1 = T.tensor_view(
                                        T.ptr_of(value_4[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_13.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_9 = T.tensor_view(
                                        T.ptr_of(value_2[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_13.d0, 8, 16 @ threads_13.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_2,
                                        rhs_frame_1,
                                        dst_frame_9,
                                        kind=BinaryKind.MUL,
                                    )
                                tile_5 = T.tensor_view(
                                    T.ptr_of(gamma_q[0:0 + 256]),
                                    layout=(1, 1, 16, 16),
                                    shape=(1, 1, 16, 16),
                                )
                                with scope_4 as threads_16:
                                    dst_frame_10 = T.tensor_view(
                                        T.ptr_of(value_9[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_16.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(tile_5, dst_frame_10)
                                    src_frame_7 = T.tensor_view(
                                        T.ptr_of(value_9[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_16.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_11 = T.tensor_view(
                                        T.ptr_of(value_8[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_16.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_frame_7, dst_frame_11, dtype="f32")
                                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_1, 1.0)
                                with scope_4 as threads_18:
                                    lhs_frame_3 = T.tensor_view(
                                        T.ptr_of(value_8[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_18.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_12 = T.tensor_view(
                                        T.ptr_of(gw[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_18.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_3,
                                        scalar_1,
                                        dst_frame_12,
                                        kind=BinaryKind.ADD,
                                    )
                                    lhs_frame_4 = T.tensor_view(
                                        T.ptr_of(value_2[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_18.d0, 8, 16 @ threads_18.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame_2 = T.tensor_view(
                                        T.ptr_of(gw[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_18.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_13 = T.tensor_view(
                                        T.ptr_of(value_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_18.d0, 8, 16 @ threads_18.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_4,
                                        rhs_frame_2,
                                        dst_frame_13,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_8 = T.tensor_view(
                                        T.ptr_of(value_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_18.d0, 8, 16 @ threads_18.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_14 = T.tensor_view(
                                        T.ptr_of(nrm[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_18.d0, 8, 16 @ threads_18.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_frame_8, dst_frame_14, dtype="bf16")
                                tile_6 = T.tensor_view(
                                    T.ptr_of(nrm[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                    layout=(16, 8, 16, 16),
                                    shape=(128, 256),
                                )
                                with scope_4 as threads_21:
                                    T.copy(tile_6, cs_1)
                                tile_7 = T.tensor_view(
                                    T.ptr_of(cs_1[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                tile_8 = T.tensor_view(
                                    T.ptr_of(tile_7[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_22:
                                    dst_frame_15 = T.tensor_view(
                                        T.ptr_of(value_14[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_22.d0, 8, 16 @ threads_22.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_8, dst_frame_15)
                                    src_frame_9 = T.tensor_view(
                                        T.ptr_of(value_14[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_22.d0, 8, 16 @ threads_22.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_16 = T.tensor_view(
                                        T.ptr_of(lo[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_22.d0, 8, 16 @ threads_22.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_9, dst_frame_16, dtype="f32")
                                tile_9 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 0:0 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_10 = T.tensor_view(
                                    T.ptr_of(tile_9[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_24:
                                    dst_frame_17 = T.tensor_view(
                                        T.ptr_of(value_15[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_10, dst_frame_17)
                                    src_frame_10 = T.tensor_view(
                                        T.ptr_of(value_15[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_18 = T.tensor_view(
                                        T.ptr_of(cl[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_10, dst_frame_18, dtype="f32")
                                    lhs_frame_5 = T.tensor_view(
                                        T.ptr_of(lo[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_3 = T.tensor_view(
                                        T.ptr_of(cl[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_19 = T.tensor_view(
                                        T.ptr_of(value_13[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_5,
                                        rhs_frame_3,
                                        dst_frame_19,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_11 = T.tensor_view(
                                        T.ptr_of(value_13[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_20 = T.tensor_view(
                                        T.ptr_of(value_12[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_11, dst_frame_20, dtype="bf16")
                                    src_frame_12 = T.tensor_view(
                                        T.ptr_of(value_12[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_21 = T.tensor_view(
                                        T.ptr_of(value_11[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_24.d0, 8, 16 @ threads_24.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_12, dst_frame_21, dtype="f32")
                                tile_11 = T.tensor_view(
                                    T.ptr_of(cs_1[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                tile_12 = T.tensor_view(
                                    T.ptr_of(tile_11[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_29:
                                    dst_frame_22 = T.tensor_view(
                                        T.ptr_of(value_20[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_29.d0, 8, 16 @ threads_29.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_12, dst_frame_22)
                                    src_frame_13 = T.tensor_view(
                                        T.ptr_of(value_20[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_29.d0, 8, 16 @ threads_29.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_23 = T.tensor_view(
                                        T.ptr_of(hi[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_29.d0, 8, 16 @ threads_29.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_13, dst_frame_23, dtype="f32")
                                    src_frame_14 = T.tensor_view(
                                        T.ptr_of(hi[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_29.d0, 8, 16 @ threads_29.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_24 = T.tensor_view(
                                        T.ptr_of(value_19[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_29.d0, 8, 16 @ threads_29.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.unary(src_frame_14, dst_frame_24, kind=UnaryKind.NEG)
                                tile_13 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 0:0 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_14 = T.tensor_view(
                                    T.ptr_of(tile_13[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_32:
                                    dst_frame_25 = T.tensor_view(
                                        T.ptr_of(value_21[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_14, dst_frame_25)
                                    src_frame_15 = T.tensor_view(
                                        T.ptr_of(value_21[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_26 = T.tensor_view(
                                        T.ptr_of(sl[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_15, dst_frame_26, dtype="f32")
                                    lhs_frame_6 = T.tensor_view(
                                        T.ptr_of(value_19[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_4 = T.tensor_view(
                                        T.ptr_of(sl[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_27 = T.tensor_view(
                                        T.ptr_of(value_18[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_6,
                                        rhs_frame_4,
                                        dst_frame_27,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_16 = T.tensor_view(
                                        T.ptr_of(value_18[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_28 = T.tensor_view(
                                        T.ptr_of(value_17[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_16, dst_frame_28, dtype="bf16")
                                    src_frame_17 = T.tensor_view(
                                        T.ptr_of(value_17[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_29 = T.tensor_view(
                                        T.ptr_of(value_16[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_17, dst_frame_29, dtype="f32")
                                    lhs_frame_7 = T.tensor_view(
                                        T.ptr_of(value_11[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_5 = T.tensor_view(
                                        T.ptr_of(value_16[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_30 = T.tensor_view(
                                        T.ptr_of(value_10[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_7,
                                        rhs_frame_5,
                                        dst_frame_30,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_18 = T.tensor_view(
                                        T.ptr_of(value_10[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_31 = T.tensor_view(
                                        T.ptr_of(rlo[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_32.d0, 8, 16 @ threads_32.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_18, dst_frame_31, dtype="bf16")
                                tile_15 = T.tensor_view(
                                    T.ptr_of(rlo[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                    layout=(16, 8, 16, 2),
                                    shape=(128, 32),
                                )
                                window = T.tensor_view(
                                    T.ptr_of(cs_1[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(tile_15, window)
                                tile_16 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 32:32 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_17 = T.tensor_view(
                                    T.ptr_of(tile_16[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_39:
                                    dst_frame_32 = T.tensor_view(
                                        T.ptr_of(value_26[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_17, dst_frame_32)
                                    src_frame_19 = T.tensor_view(
                                        T.ptr_of(value_26[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_33 = T.tensor_view(
                                        T.ptr_of(ch[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_19, dst_frame_33, dtype="f32")
                                    lhs_frame_8 = T.tensor_view(
                                        T.ptr_of(hi[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_6 = T.tensor_view(
                                        T.ptr_of(ch[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_34 = T.tensor_view(
                                        T.ptr_of(value_25[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_8,
                                        rhs_frame_6,
                                        dst_frame_34,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_20 = T.tensor_view(
                                        T.ptr_of(value_25[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_35 = T.tensor_view(
                                        T.ptr_of(value_24[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_20, dst_frame_35, dtype="bf16")
                                    src_frame_21 = T.tensor_view(
                                        T.ptr_of(value_24[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_36 = T.tensor_view(
                                        T.ptr_of(value_23[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_39.d0, 8, 16 @ threads_39.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_21, dst_frame_36, dtype="f32")
                                tile_18 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 32:32 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_19 = T.tensor_view(
                                    T.ptr_of(tile_18[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_4 as threads_44:
                                    dst_frame_37 = T.tensor_view(
                                        T.ptr_of(value_30[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_19, dst_frame_37)
                                    src_frame_22 = T.tensor_view(
                                        T.ptr_of(value_30[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_38 = T.tensor_view(
                                        T.ptr_of(sh[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_22, dst_frame_38, dtype="f32")
                                    lhs_frame_9 = T.tensor_view(
                                        T.ptr_of(lo[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_7 = T.tensor_view(
                                        T.ptr_of(sh[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_39 = T.tensor_view(
                                        T.ptr_of(value_29[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_9,
                                        rhs_frame_7,
                                        dst_frame_39,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_23 = T.tensor_view(
                                        T.ptr_of(value_29[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_40 = T.tensor_view(
                                        T.ptr_of(value_28[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_23, dst_frame_40, dtype="bf16")
                                    src_frame_24 = T.tensor_view(
                                        T.ptr_of(value_28[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_41 = T.tensor_view(
                                        T.ptr_of(value_27[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_24, dst_frame_41, dtype="f32")
                                    lhs_frame_10 = T.tensor_view(
                                        T.ptr_of(value_23[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_8 = T.tensor_view(
                                        T.ptr_of(value_27[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_42 = T.tensor_view(
                                        T.ptr_of(value_22[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_10,
                                        rhs_frame_8,
                                        dst_frame_42,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_25 = T.tensor_view(
                                        T.ptr_of(value_22[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_43 = T.tensor_view(
                                        T.ptr_of(rhi[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_44.d0, 8, 16 @ threads_44.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_25, dst_frame_43, dtype="bf16")
                                tile_20 = T.tensor_view(
                                    T.ptr_of(rhi[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                    layout=(16, 8, 16, 2),
                                    shape=(128, 32),
                                )
                                window_1 = T.tensor_view(
                                    T.ptr_of(cs_1[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(tile_20, window_1)
                            T.copy(cs_1, cs)
                    for _k0 in range(47, bn - (56 * (bn // 56)), 1048576):
                        cs_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        for _k1 in range(bn - (56 * (bn // 56)), 52, 1048576):
                            with Mesh(
                                scope[:2], layout=(16, 16), names=("d0", "d1")
                            ) as scope_5:
                                tile_21 = T.tensor_view(
                                    T.ptr_of(cs[0:0 + 128, 0:0 + 256]),
                                    layout=(16, 8, 16, 16) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 16),
                                )
                                with scope_5 as threads_51:
                                    dst_frame_44 = T.tensor_view(
                                        T.ptr_of(value_33[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(tile_21, dst_frame_44)
                                    src_frame_26 = T.tensor_view(
                                        T.ptr_of(value_33[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_45 = T.tensor_view(
                                        T.ptr_of(x32_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_frame_26, dst_frame_45, dtype="f32")
                                    lhs_frame_11 = T.tensor_view(
                                        T.ptr_of(x32_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame_9 = T.tensor_view(
                                        T.ptr_of(x32_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_46 = T.tensor_view(
                                        T.ptr_of(value_37[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_11,
                                        rhs_frame_9,
                                        dst_frame_46,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_27 = T.tensor_view(
                                        T.ptr_of(value_37[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_47 = T.tensor_view(
                                        T.ptr_of(sq_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(src_frame_27, dst_frame_47)
                                    src_frame_28 = T.tensor_view(
                                        T.ptr_of(sq_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=((16 @ threads_51.d0, 8, 16 @ threads_51.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_48 = T.tensor_view(
                                        T.ptr_of(value_36[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=((16 @ threads_51.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.reduce(
                                        src_frame_28,
                                        dst_frame_48,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    src_frame_29 = T.tensor_view(
                                        T.ptr_of(value_36[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=((16 @ threads_51.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_49 = T.tensor_view(
                                        T.ptr_of(ms_1[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_51.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(src_frame_29, dst_frame_49)
                                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_2, 1e-06)
                                with scope_5 as threads_57:
                                    lhs_frame_12 = T.tensor_view(
                                        T.ptr_of(ms_1[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_57.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_50 = T.tensor_view(
                                        T.ptr_of(value_35[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_57.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.binary(
                                        lhs_frame_12,
                                        scalar_2,
                                        dst_frame_50,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_30 = T.tensor_view(
                                        T.ptr_of(value_35[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_57.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_51 = T.tensor_view(
                                        T.ptr_of(value_34[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_57.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.unary(src_frame_30, dst_frame_51, kind=UnaryKind.RSQRT)
                                    lhs_frame_13 = T.tensor_view(
                                        T.ptr_of(x32_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_57.d0, 8, 16 @ threads_57.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame_10 = T.tensor_view(
                                        T.ptr_of(value_34[0:0 + 16, 0:0 + 8, 0:0 + 1, 0:0 + 1]),
                                        layout=(16 @ threads_57.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    dst_frame_52 = T.tensor_view(
                                        T.ptr_of(value_32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_57.d0, 8, 16 @ threads_57.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_13,
                                        rhs_frame_10,
                                        dst_frame_52,
                                        kind=BinaryKind.MUL,
                                    )
                                tile_22 = T.tensor_view(
                                    T.ptr_of(gamma_k[0:0 + 256]),
                                    layout=(1, 1, 16, 16),
                                    shape=(1, 1, 16, 16),
                                )
                                with scope_5 as threads_60:
                                    dst_frame_53 = T.tensor_view(
                                        T.ptr_of(value_39[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_60.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(tile_22, dst_frame_53)
                                    src_frame_31 = T.tensor_view(
                                        T.ptr_of(value_39[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_60.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_54 = T.tensor_view(
                                        T.ptr_of(value_38[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_60.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.cast(src_frame_31, dst_frame_54, dtype="f32")
                                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_3, 1.0)
                                with scope_5 as threads_62:
                                    lhs_frame_14 = T.tensor_view(
                                        T.ptr_of(value_38[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_62.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_55 = T.tensor_view(
                                        T.ptr_of(gw_1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_62.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_14,
                                        scalar_3,
                                        dst_frame_55,
                                        kind=BinaryKind.ADD,
                                    )
                                    lhs_frame_15 = T.tensor_view(
                                        T.ptr_of(value_32[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_62.d0, 8, 16 @ threads_62.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    rhs_frame_11 = T.tensor_view(
                                        T.ptr_of(gw_1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 16]),
                                        layout=(1, 1, 16 @ threads_62.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    dst_frame_56 = T.tensor_view(
                                        T.ptr_of(value_31[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_62.d0, 8, 16 @ threads_62.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.binary(
                                        lhs_frame_15,
                                        rhs_frame_11,
                                        dst_frame_56,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_32 = T.tensor_view(
                                        T.ptr_of(value_31[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_62.d0, 8, 16 @ threads_62.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    dst_frame_57 = T.tensor_view(
                                        T.ptr_of(nrm_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                        layout=(16 @ threads_62.d0, 8, 16 @ threads_62.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.cast(src_frame_32, dst_frame_57, dtype="bf16")
                                tile_23 = T.tensor_view(
                                    T.ptr_of(nrm_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 16]),
                                    layout=(16, 8, 16, 16),
                                    shape=(128, 256),
                                )
                                with scope_5 as threads_65:
                                    T.copy(tile_23, cs_2)
                                tile_24 = T.tensor_view(
                                    T.ptr_of(cs_2[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                tile_25 = T.tensor_view(
                                    T.ptr_of(tile_24[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_66:
                                    dst_frame_58 = T.tensor_view(
                                        T.ptr_of(value_44[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_66.d0, 8, 16 @ threads_66.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_25, dst_frame_58)
                                    src_frame_33 = T.tensor_view(
                                        T.ptr_of(value_44[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_66.d0, 8, 16 @ threads_66.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_59 = T.tensor_view(
                                        T.ptr_of(lo_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_66.d0, 8, 16 @ threads_66.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_33, dst_frame_59, dtype="f32")
                                tile_26 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 0:0 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_27 = T.tensor_view(
                                    T.ptr_of(tile_26[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_68:
                                    dst_frame_60 = T.tensor_view(
                                        T.ptr_of(value_45[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_27, dst_frame_60)
                                    src_frame_34 = T.tensor_view(
                                        T.ptr_of(value_45[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_61 = T.tensor_view(
                                        T.ptr_of(cl_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_34, dst_frame_61, dtype="f32")
                                    lhs_frame_16 = T.tensor_view(
                                        T.ptr_of(lo_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_12 = T.tensor_view(
                                        T.ptr_of(cl_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_62 = T.tensor_view(
                                        T.ptr_of(value_43[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_16,
                                        rhs_frame_12,
                                        dst_frame_62,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_35 = T.tensor_view(
                                        T.ptr_of(value_43[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_63 = T.tensor_view(
                                        T.ptr_of(value_42[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_35, dst_frame_63, dtype="bf16")
                                    src_frame_36 = T.tensor_view(
                                        T.ptr_of(value_42[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_64 = T.tensor_view(
                                        T.ptr_of(value_41[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_68.d0, 8, 16 @ threads_68.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_36, dst_frame_64, dtype="f32")
                                tile_28 = T.tensor_view(
                                    T.ptr_of(cs_2[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                tile_29 = T.tensor_view(
                                    T.ptr_of(tile_28[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_73:
                                    dst_frame_65 = T.tensor_view(
                                        T.ptr_of(value_50[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_73.d0, 8, 16 @ threads_73.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_29, dst_frame_65)
                                    src_frame_37 = T.tensor_view(
                                        T.ptr_of(value_50[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_73.d0, 8, 16 @ threads_73.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_66 = T.tensor_view(
                                        T.ptr_of(hi_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_73.d0, 8, 16 @ threads_73.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_37, dst_frame_66, dtype="f32")
                                    src_frame_38 = T.tensor_view(
                                        T.ptr_of(hi_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_73.d0, 8, 16 @ threads_73.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_67 = T.tensor_view(
                                        T.ptr_of(value_49[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_73.d0, 8, 16 @ threads_73.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.unary(src_frame_38, dst_frame_67, kind=UnaryKind.NEG)
                                tile_30 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 0:0 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_31 = T.tensor_view(
                                    T.ptr_of(tile_30[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_76:
                                    dst_frame_68 = T.tensor_view(
                                        T.ptr_of(value_51[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_31, dst_frame_68)
                                    src_frame_39 = T.tensor_view(
                                        T.ptr_of(value_51[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_69 = T.tensor_view(
                                        T.ptr_of(sl_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_39, dst_frame_69, dtype="f32")
                                    lhs_frame_17 = T.tensor_view(
                                        T.ptr_of(value_49[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_13 = T.tensor_view(
                                        T.ptr_of(sl_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_70 = T.tensor_view(
                                        T.ptr_of(value_48[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_17,
                                        rhs_frame_13,
                                        dst_frame_70,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_40 = T.tensor_view(
                                        T.ptr_of(value_48[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_71 = T.tensor_view(
                                        T.ptr_of(value_47[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_40, dst_frame_71, dtype="bf16")
                                    src_frame_41 = T.tensor_view(
                                        T.ptr_of(value_47[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_72 = T.tensor_view(
                                        T.ptr_of(value_46[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_41, dst_frame_72, dtype="f32")
                                    lhs_frame_18 = T.tensor_view(
                                        T.ptr_of(value_41[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_14 = T.tensor_view(
                                        T.ptr_of(value_46[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_73 = T.tensor_view(
                                        T.ptr_of(value_40[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_18,
                                        rhs_frame_14,
                                        dst_frame_73,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_42 = T.tensor_view(
                                        T.ptr_of(value_40[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_74 = T.tensor_view(
                                        T.ptr_of(rlo_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_76.d0, 8, 16 @ threads_76.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_42, dst_frame_74, dtype="bf16")
                                tile_32 = T.tensor_view(
                                    T.ptr_of(rlo_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                    layout=(16, 8, 16, 2),
                                    shape=(128, 32),
                                )
                                window_2 = T.tensor_view(
                                    T.ptr_of(cs_2[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(tile_32, window_2)
                                tile_33 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 32:32 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_34 = T.tensor_view(
                                    T.ptr_of(tile_33[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_83:
                                    dst_frame_75 = T.tensor_view(
                                        T.ptr_of(value_56[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_34, dst_frame_75)
                                    src_frame_43 = T.tensor_view(
                                        T.ptr_of(value_56[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_76 = T.tensor_view(
                                        T.ptr_of(ch_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_43, dst_frame_76, dtype="f32")
                                    lhs_frame_19 = T.tensor_view(
                                        T.ptr_of(hi_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_15 = T.tensor_view(
                                        T.ptr_of(ch_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_77 = T.tensor_view(
                                        T.ptr_of(value_55[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_19,
                                        rhs_frame_15,
                                        dst_frame_77,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_44 = T.tensor_view(
                                        T.ptr_of(value_55[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_78 = T.tensor_view(
                                        T.ptr_of(value_54[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_44, dst_frame_78, dtype="bf16")
                                    src_frame_45 = T.tensor_view(
                                        T.ptr_of(value_54[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_79 = T.tensor_view(
                                        T.ptr_of(value_53[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_83.d0, 8, 16 @ threads_83.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_45, dst_frame_79, dtype="f32")
                                tile_35 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 128, 32:32 + 32]),
                                    layout=((128, 32), (64, 1)),
                                    shape=(128, 32),
                                )
                                tile_36 = T.tensor_view(
                                    T.ptr_of(tile_35[0:0 + 128, 0:0 + 32]),
                                    layout=(16, 8, 16, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                with scope_5 as threads_88:
                                    dst_frame_80 = T.tensor_view(
                                        T.ptr_of(value_60[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(tile_36, dst_frame_80)
                                    src_frame_46 = T.tensor_view(
                                        T.ptr_of(value_60[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_81 = T.tensor_view(
                                        T.ptr_of(sh_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_46, dst_frame_81, dtype="f32")
                                    lhs_frame_20 = T.tensor_view(
                                        T.ptr_of(lo_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_16 = T.tensor_view(
                                        T.ptr_of(sh_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_82 = T.tensor_view(
                                        T.ptr_of(value_59[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_20,
                                        rhs_frame_16,
                                        dst_frame_82,
                                        kind=BinaryKind.MUL,
                                    )
                                    src_frame_47 = T.tensor_view(
                                        T.ptr_of(value_59[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_83 = T.tensor_view(
                                        T.ptr_of(value_58[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_47, dst_frame_83, dtype="bf16")
                                    src_frame_48 = T.tensor_view(
                                        T.ptr_of(value_58[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_84 = T.tensor_view(
                                        T.ptr_of(value_57[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_48, dst_frame_84, dtype="f32")
                                    lhs_frame_21 = T.tensor_view(
                                        T.ptr_of(value_53[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    rhs_frame_17 = T.tensor_view(
                                        T.ptr_of(value_57[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_85 = T.tensor_view(
                                        T.ptr_of(value_52[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.binary(
                                        lhs_frame_21,
                                        rhs_frame_17,
                                        dst_frame_85,
                                        kind=BinaryKind.ADD,
                                    )
                                    src_frame_49 = T.tensor_view(
                                        T.ptr_of(value_52[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    dst_frame_86 = T.tensor_view(
                                        T.ptr_of(rhi_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                        layout=(16 @ threads_88.d0, 8, 16 @ threads_88.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.cast(src_frame_49, dst_frame_86, dtype="bf16")
                                tile_37 = T.tensor_view(
                                    T.ptr_of(rhi_1[0:0 + 16, 0:0 + 8, 0:0 + 16, 0:0 + 2]),
                                    layout=(16, 8, 16, 2),
                                    shape=(128, 32),
                                )
                                window_3 = T.tensor_view(
                                    T.ptr_of(cs_2[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(tile_37, window_3)
                            T.copy(cs_2, cs)
                    with scope[:1, :32] as scope_6:
                        with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_95:
                            window_4 = T.tensor_view(
                                T.ptr_of(out_2_1[(128 * mi) + 6144:(128 * mi) + 6144 + 128, 256 * bn:256 * bn + 256]),
                                layout=((128, 256), (14336, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(cs, window_4)
        T.copy(out_2_1, out_1)
