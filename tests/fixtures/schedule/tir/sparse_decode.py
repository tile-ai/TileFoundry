# analysis target=nvidia.h200_sxm module=SparseDecode function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r17.00KB/w16.00KB@logical,r17.00KB/w16.00KB@total,r17.00KB/w16.00KB@cta,r17.00KB/w8.06KB@thread;rmem:r189.60KB/w153.52KB@logical,r189.60KB/w153.52KB@total,r189.60KB/w153.52KB@cta,r6.54KB/w2.23KB@thread;smem:r40.00KB/w48.00KB@logical,r40.00KB/w48.00KB@total,r40.00KB/w48.00KB@cta,r18.75KB/w48.00KB@thread footprint=idx:512B;kv:128B;q:128B;v45:151:8.00KB footprint-precision=exact peak=gmem:72.50KB;rmem:32.25KB;smem:16.00KB persistent=gmem:64.50KB
#   buffer=kv holds=264B time=r space=none reuse=7.88KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    q: Tensor[(1, 2, 128, 64), "bf16"], kv: Tensor[(1, 256, 1, 64), "bf16"], idx: Tensor[(64,), "i64"], out: Tensor[(64, 64), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 256),), layout=(2, 128), names=("role", "participant")
        ) as scope:
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
                        value = T.tensor_view(T.ptr_of(kv), layout=(256, 64), shape=(256, 64))
                        ks_2 = T.tensor_view(
                            0,
                            dtype="bf16",
                            storage="smem",
                            layout=(64, 64) | Swizzle(3, 4, 3),
                            shape=(64, 64),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(ks_2[r:r + 1, 0:0 + 1]),
                            layout=(1, 64) | Swizzle(3, 4, 3),
                            shape=(1, 64),
                        )
                        value_2 = T.tensor_view(
                            T.ptr_of(idx[r:r + 1]), layout=((1,), (0,)), shape=(1,)
                        )
                        T.copy_async(value, value_1, value_2, fill=0)
                        value_3 = T.tensor_view(
                            T.ptr_of(q[0:0 + 1, 1:1 + 1, 32:32 + 1, 0:0 + 1]),
                            layout=((1, 1, 64, 64), (0, 0, 64, 1)),
                            shape=(1, 1, 64, 64),
                        )
                        value_4 = T.tensor_view(
                            T.ptr_of(value_3), layout=(64, 64), shape=(64, 64)
                        )
                        value_5 = T.tensor_view(
                            T.ptr_of(value_4[r:r + 1, 0:0 + 1]),
                            layout=((1, 64), (0, 1)),
                            shape=(1, 64),
                        )
                        qs_2 = T.tensor_view(
                            8192,
                            dtype="bf16",
                            storage="smem",
                            layout=(64, 64) | Swizzle(3, 4, 3),
                            shape=(64, 64),
                        )
                        value_6 = T.tensor_view(
                            T.ptr_of(qs_2[r:r + 1, 0:0 + 1]),
                            layout=(1, 64) | Swizzle(3, 4, 3),
                            shape=(1, 64),
                        )
                        T.copy_async(value_5, value_6)
            T.fill(out, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_3:
                o = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(o, 0.0)
                ii_dst = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", (64,), "rmem"])
                T.copy(idx, ii_dst)
                ii = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", (64,), "rmem"])
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar, 0)
                T.binary(ii_dst, scalar, ii, kind=BinaryKind.ADD)
                value_7 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_1, 0)
                T.binary(ii, scalar_1, value_7, kind=BinaryKind.GE)
                value_8 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_2, 256)
                T.binary(ii, scalar_2, value_8, kind=BinaryKind.LT)
                live = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", (64,), "rmem"])
                T.binary(value_7, value_8, live, kind=BinaryKind.AND)
                s = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(s, 0.0)
                acc_view = T.tensor_view(
                    T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                qs_3 = T.tensor_view(
                    8192,
                    dtype="bf16",
                    storage="smem",
                    layout=(64, 64) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_9 = T.tensor_view(
                    T.ptr_of(qs_3), layout=(64, 64) | Swizzle(3, 4, 3), shape=(64, 64)
                )
                lhs_view = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64, 0:0 + 16]),
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
                value_10 = T.tensor_view(
                    T.ptr_of(ks_3),
                    layout=((64, 64), (1, 64)) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                rhs_view = T.tensor_view(
                    T.ptr_of(value_10[0:0 + 16, 0:0 + 64]),
                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view,
                    lhs_view,
                    rhs_view,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                )
                acc_view_1 = T.tensor_view(
                    T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_1 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64, 16:16 + 16]),
                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(64, 16),
                )
                rhs_view_1 = T.tensor_view(
                    T.ptr_of(value_10[16:16 + 16, 0:0 + 64]),
                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_1,
                    lhs_view_1,
                    rhs_view_1,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                )
                acc_view_2 = T.tensor_view(
                    T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_2 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64, 32:32 + 16]),
                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(64, 16),
                )
                rhs_view_2 = T.tensor_view(
                    T.ptr_of(value_10[32:32 + 16, 0:0 + 64]),
                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_2,
                    lhs_view_2,
                    rhs_view_2,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                )
                acc_view_3 = T.tensor_view(
                    T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_3 = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64, 48:48 + 16]),
                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(64, 16),
                )
                rhs_view_3 = T.tensor_view(
                    T.ptr_of(value_10[48:48 + 16, 0:0 + 64]),
                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_3,
                    lhs_view_3,
                    rhs_view_3,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                )
                s_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, -1e999)
                value_11 = T.tensor_view(T.ptr_of(live), layout=(64,), shape=(64,))
                value_12 = T.tensor_view(
                    T.ptr_of(value_11), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                lhs_view_4 = T.tensor_view(
                    T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.where(value_12, lhs_view_4, scalar_3, s_1)
                peak = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(s_1, peak, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(s_1, peak, value_13, kind=BinaryKind.SUB)
                p = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_13, p, kind=UnaryKind.EXP)
                denom = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(p, denom, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                value_14 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(p, denom, value_14, kind=BinaryKind.DIV)
                p_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_14, p_1, dtype="bf16")
                pr_dst = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.copy(p_1, pr_dst)
                acc_view_4 = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_5 = T.tensor_view(
                    T.ptr_of(pr_dst[0:0 + 64, 0:0 + 16]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                value_15 = T.tensor_view(
                    T.ptr_of(ks_3), layout=(64, 64) | Swizzle(3, 4, 3), shape=(64, 64)
                )
                rhs_view_4 = T.tensor_view(
                    T.ptr_of(value_15[0:0 + 16, 0:0 + 64]),
                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_4,
                    lhs_view_5,
                    rhs_view_4,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                )
                acc_view_5 = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_6 = T.tensor_view(
                    T.ptr_of(pr_dst[0:0 + 64, 16:16 + 16]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                rhs_view_5 = T.tensor_view(
                    T.ptr_of(value_15[16:16 + 16, 0:0 + 64]),
                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_5,
                    lhs_view_6,
                    rhs_view_5,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                )
                acc_view_6 = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_7 = T.tensor_view(
                    T.ptr_of(pr_dst[0:0 + 64, 32:32 + 16]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                rhs_view_6 = T.tensor_view(
                    T.ptr_of(value_15[32:32 + 16, 0:0 + 64]),
                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_6,
                    lhs_view_7,
                    rhs_view_6,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                )
                acc_view_7 = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                lhs_view_8 = T.tensor_view(
                    T.ptr_of(pr_dst[0:0 + 64, 48:48 + 16]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                rhs_view_7 = T.tensor_view(
                    T.ptr_of(value_15[48:48 + 16, 0:0 + 64]),
                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                    shape=(16, 64),
                )
                T.tiled_mma(
                    acc_view_7,
                    lhs_view_8,
                    rhs_view_7,
                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                )
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                src_view = T.tensor_view(
                    T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                    layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(src_view, result, dtype="bf16")
                T.copy(result, out)
