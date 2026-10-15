# analysis target=nvidia.h200_sxm module=SparseDecode function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r17.00KB/w16.00KB@logical,r17.00KB/w16.00KB@total,r17.00KB/w16.00KB@cta,r17.00KB/w8.06KB@thread;rmem:r189.60KB/w153.52KB@logical,r189.60KB/w153.52KB@total,r189.60KB/w153.52KB@cta,r6.51KB/w2.20KB@thread;smem:r40.00KB/w48.00KB@logical,r40.00KB/w48.00KB@total,r40.00KB/w48.00KB@cta,r18.75KB/w48.00KB@thread footprint=kv:128B;idx:512B;q:128B;v48:168:8.00KB footprint-precision=exact peak=gmem:72.50KB;rmem:32.25KB;smem:16.00KB persistent=gmem:64.50KB
#   buffer=kv holds=264B time=r space=none reuse=7.88KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    q: Tensor[(1, 2, 128, 64), "bf16"], kv: Tensor[(1, 256, 1, 64), "bf16"], idx: Tensor[(64,), "i64"], out: Tensor[(64, 64), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            with scope[:1, :32] as scope_1:
                qs = T.tensor_view(
                    8192,
                    dtype="bf16",
                    storage="smem",
                    layout=(64, 64) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.fill(qs, 0.0)
                ks = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=(64, 64) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.fill(ks, 0.0)
                for r in range(0, 64, 1):
                    with scope_1 as scope_2:
                        tile = T.tensor_view(
                            T.ptr_of(kv[0:0 + 1, 0:0 + 256, 0:0 + 1, 0:0 + 64]),
                            layout=(256, 64),
                            shape=(256, 64),
                        )
                        ks_2 = T.tensor_view(
                            0,
                            dtype="bf16",
                            storage="smem",
                            layout=(64, 64) | Swizzle(3, 4, 3),
                            shape=(64, 64),
                        )
                        window = T.tensor_view(
                            T.ptr_of(ks_2[r:r + 1, 0:0 + 64]),
                            layout=(1, 64) | Swizzle(3, 4, 3),
                            shape=(1, 64),
                        )
                        tile_1 = T.tensor_view(T.ptr_of(idx[r:r + 1]), layout=(1,), shape=(1,))
                        T.copy_async(tile, window, tile_1, fill=0)
                        tile_2 = T.tensor_view(
                            T.ptr_of(q[0:0 + 1, 1:1 + 1, 32:32 + 64, 0:0 + 64]),
                            layout=((1, 1, 64, 64), (16384, 8192, 64, 1)),
                            shape=(1, 1, 64, 64),
                        )
                        tile_3 = T.tensor_view(
                            T.ptr_of(tile_2[0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 64]),
                            layout=(64, 64),
                            shape=(64, 64),
                        )
                        tile_4 = T.tensor_view(
                            T.ptr_of(tile_3[r:r + 1, 0:0 + 64]), layout=(1, 64), shape=(1, 64)
                        )
                        qs_2 = T.tensor_view(
                            8192,
                            dtype="bf16",
                            storage="smem",
                            layout=(64, 64) | Swizzle(3, 4, 3),
                            shape=(64, 64),
                        )
                        window_1 = T.tensor_view(
                            T.ptr_of(qs_2[r:r + 1, 0:0 + 64]),
                            layout=(1, 64) | Swizzle(3, 4, 3),
                            shape=(1, 64),
                        )
                        T.copy_async(tile_4, window_1)
            T.fill(out, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                o = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(o, 0.0)
                pr_dst = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        "rmem",
                    ]
                )
                ii_dst = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", (64,), "rmem"])
                T.copy(idx, ii_dst)
                ii = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", (64,), "rmem"])
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar, 0)
                T.binary(ii_dst, scalar, ii, kind=BinaryKind.ADD)
                value = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_1, 0)
                T.binary(ii, scalar_1, value, kind=BinaryKind.GE)
                value_1 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_2, 256)
                T.binary(ii, scalar_2, value_1, kind=BinaryKind.LT)
                live = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                T.binary(value, value_1, live, kind=BinaryKind.AND)
                s = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(s, 0.0)
                for o_m in range(0, 64, 64):
                    for o_n in range(0, 64, 64):
                        for o_k in range(0, 64, 64):
                            qs_3 = T.tensor_view(
                                8192,
                                dtype="bf16",
                                storage="smem",
                                layout=(64, 64) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            tile_5 = T.tensor_view(
                                T.ptr_of(qs_3[0:0 + 64, 0:0 + 64]),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view = T.tensor_view(
                                T.ptr_of(tile_5[o_m:o_m + 64, o_k:o_k + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            ks_3 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=(64, 64) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            tile_6 = T.tensor_view(
                                T.ptr_of(ks_3[0:0 + 64, 0:0 + 64]),
                                layout=Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view = T.tensor_view(
                                T.ptr_of(tile_6[o_k:o_k + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                s,
                                lhs_view,
                                rhs_view,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_1 = T.tensor_view(
                                T.ptr_of(tile_5[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_1 = T.tensor_view(
                                T.ptr_of(tile_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                s,
                                lhs_view_1,
                                rhs_view_1,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_2 = T.tensor_view(
                                T.ptr_of(tile_5[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_2 = T.tensor_view(
                                T.ptr_of(tile_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                s,
                                lhs_view_2,
                                rhs_view_2,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_3 = T.tensor_view(
                                T.ptr_of(tile_5[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_3 = T.tensor_view(
                                T.ptr_of(tile_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                s,
                                lhs_view_3,
                                rhs_view_3,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                s_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, -1e999)
                tile_7 = T.tensor_view(
                    T.ptr_of(live[0:0 + 64]), layout=Layout(((8, 8),), ((8, 1),)), shape=(64,)
                )
                tile_8 = T.tensor_view(
                    T.ptr_of(tile_7[0:0 + 64]), layout=(1, 64), shape=(1, 64)
                )
                T.where(tile_8, s, scalar_3, s_1)
                peak = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(s_1, peak, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                value_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(s_1, peak, value_2, kind=BinaryKind.SUB)
                p = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_2, p, kind=UnaryKind.EXP)
                denom = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(p, denom, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                value_3 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(p, denom, value_3, kind=BinaryKind.DIV)
                p_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_3, p_1, dtype="bf16")
                tile_9 = T.tensor_view(
                    T.ptr_of(p_1[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2), (16, 8, 1, 512, 64, 128)),
                    shape=(64, 64),
                )
                T.copy(tile_9, pr_dst)
                for o_m_1 in range(0, 64, 64):
                    for o_n_1 in range(0, 64, 64):
                        for o_k_1 in range(0, 64, 16):
                            lhs_view_4 = T.tensor_view(
                                T.ptr_of(pr_dst[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                shape=(64, 16),
                            )
                            ks_4 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=(64, 64) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            tile_10 = T.tensor_view(
                                T.ptr_of(ks_4[0:0 + 64, 0:0 + 64]),
                                layout=Layout(((8, 8), (8, 8)), ((64, 512), (1, 8))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            tile_11 = T.tensor_view(
                                T.ptr_of(tile_10[0:0 + 64, 0:0 + 64]),
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_4 = T.tensor_view(
                                T.ptr_of(tile_11[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                o,
                                lhs_view_4,
                                rhs_view_4,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((8 @ scope_3.d1, 4 @ scope_3.d0, 2, 8, 4 @ scope_3.d2, 2), (1, 16, 8, 512, 128, 64)),
                        "rmem",
                    ]
                )
                tile_12 = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((8 @ scope_3.d1, 4 @ scope_3.d0, 2, 8, 4 @ scope_3.d2, 2), (1, 16, 8, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(tile_12, result, dtype="bf16")
                window_2 = T.tensor_view(
                    T.ptr_of(out[0:0 + 64, 0:0 + 64]), layout=(64, 64), shape=(64, 64)
                )
                T.copy(result, window_2)
