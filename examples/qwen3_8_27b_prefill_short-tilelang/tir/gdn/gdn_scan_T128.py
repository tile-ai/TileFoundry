# analysis target=nvidia.h200_sxm module=SCHED_GDN_SCAN_T128 function=gdn_scan topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r241.50KB/w6.12MB@logical,r22.64MB/w588.00MB@total,r241.50KB/w6.12MB@cta,r241.50KB/w6.06MB@thread;rmem:r1.33MB/w1.00MB@logical,r127.66MB/w99.10MB@total,r1.33MB/w1.03MB@cta,r11.36KB/w8.34KB@thread;smem:r257.01KB/w193.50KB@logical,r24.09MB/w18.14MB@total,r257.01KB/w193.50KB@cta,r13.57KB/w177.62KB@thread footprint=fc:36.00KB;k:256.00KB;pb:384.00KB;q:256.00KB;ut:1.50MB;v101:113:16.00KB;v105:114:1.50MB;v109:116:32.00KB;v113:117:3.00MB;wb:1.50MB footprint-precision=exact peak=gmem:13.85MB;rmem:160.02KB;smem:193.50KB persistent=gmem:7.82MB
#   buffer=wb holds=8.46MB time=none space=cta.vb reuse=1.50MB fits=yes precision=exact
#   buffer=pb holds=8.46MB time=none space=cta.vb reuse=384.00KB fits=yes precision=exact
#   buffer=q holds=8.46MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=k holds=8.46MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=fc holds=8.46MB time=none space=cta.vb reuse=36.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_scan(
    q: Tensor[(128, 16, 128), "bf16"], k: Tensor[(128, 16, 128), "bf16"], ut: Tensor[(48, 2, 128, 64), "f32"], wb: Tensor[(48, 2, 2, 64, 128), "bf16"], pb: Tensor[(48, 2, 64, 64), "bf16"], fc: Tensor[(48, 2, 3, 64), "f32"], scratch: Tensor[(4096,), "f32"], scratch_1: Tensor[(8192,), "f32"], out: Tensor[(256, 48, 128), "f32"]
):
    with Mesh((Topology("cta", 96),), layout=(48, 2), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            St = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 128),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 2, 8, 4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        attrs=(S(0), S(2), S(5)),
                        mesh=Mesh((Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_1:
                T.fill(St, 0.0)
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_3_stages = (
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(49152, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_6_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_9_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_12_stages = (
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(172032, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_15_stages = (
                T.tensor_view(131072, dtype="f32", storage="smem", layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(147456, dtype="f32", storage="smem", layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_18_stages = (
                T.tensor_view(196608, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)),
                T.tensor_view(197376, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)),
            )
            value_60_stages = (
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=((64, 64), (1, 72)), shape=(64, 64)),
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=((64, 64), (1, 72)), shape=(64, 64)),
            )
            for c in range(0, 2, 1):
                with scope[1:2, :32] as scope_2:
                    value_1 = T.tensor_view(
                        T.ptr_of(wb[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(64, 128),
                        shape=(64, 128),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(value_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_1, value_2)
                    value_4 = T.tensor_view(
                        T.ptr_of(wb[cta.d0:cta.d0 + 1, c:c + 1, 1:1 + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(64, 128),
                        shape=(64, 128),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(value_3_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_4, value_5)
                    value_7 = T.tensor_view(
                        T.ptr_of(q[c * 64:c * 64 + 1, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 1]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_7, value_8)
                    value_10 = T.tensor_view(
                        T.ptr_of(k[c * 64:c * 64 + 1, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 1]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_9_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_10, value_11)
                    value_13 = T.tensor_view(
                        T.ptr_of(pb[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(64, 64),
                        shape=(64, 64),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_12_stages[c % 2]),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_13, value_14)
                    value_16 = T.tensor_view(
                        T.ptr_of(ut[cta.d0:cta.d0 + 1, c:c + 1, cta.d1 * 64:cta.d1 * 64 + 1, 0:0 + 1]),
                        layout=(64, 64),
                        shape=(64, 64),
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_16, value_17)
                    value_19 = T.tensor_view(
                        T.ptr_of(fc[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 1, 0:0 + 1]),
                        layout=(3, 64),
                        shape=(3, 64),
                    )
                    value_20 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 2]), layout=(3, 64), shape=(3, 64)
                    )
                    T.copy_async_tensor(value_19, value_20)
                St_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ShardLayout(
                            layout=Layout((4, 2, 8, 2, 8, 4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                            attrs=(S(0), S(2), S(5)),
                            mesh=Mesh((Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
                ) as scope_3:
                    value_21 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    value_22 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 2][2:2 + 1, 0:0 + 1]),
                        layout=((1, 1), (0, 0)),
                        shape=(1, 1),
                    )
                    value_23 = T.tensor_view(T.ptr_of(value_21), layout=(1, 1), shape=(1, 1))
                    T.copy(value_22, value_23)
                    value_24 = T.tensor_view(
                        T.ptr_of(St),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        shape=(64, 128),
                    )
                    T.binary(value_24, value_23, St_1, kind=BinaryKind.MUL)
                    value_25 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_26 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_27 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_28 = T.tensor_view(
                        T.ptr_of(value_26),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_27, value_28)
                    value_29 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(value_29, 0.0)
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    sbv = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_24, sbv, dtype="bf16")
                    value_31 = T.tensor_view(
                        T.ptr_of(value_30),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 128),
                    )
                    T.copy(sbv, value_31)
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 64, 64):
                            for o_k in range(0, 128, 64):
                                value_32 = T.tensor_view(
                                    T.ptr_of(value_30),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 128),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_32[o_m:o_m + 64, o_k:o_k + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_33 = T.tensor_view(
                                    T.ptr_of(value_stages[c % 2]),
                                    layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_33[o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_29,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_32[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_33[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_29,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_32[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_33[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_29,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_32[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_33[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_29,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    for o_m_1 in range(0, 64, 64):
                        for o_n_1 in range(0, 64, 64):
                            for o_k_1 in range(0, 128, 64):
                                value_34 = T.tensor_view(
                                    T.ptr_of(value_29),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                value_35 = T.tensor_view(
                                    T.ptr_of(value_30),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 128),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_35[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_36 = T.tensor_view(
                                    T.ptr_of(value_3_stages[c % 2]),
                                    layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_36[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_34,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_35[o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_36[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_34,
                                    lhs_view_5,
                                    rhs_view_5,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_35[o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_36[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_34,
                                    lhs_view_6,
                                    rhs_view_6,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_35[o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_36[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_34,
                                    lhs_view_7,
                                    rhs_view_7,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_37 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_38 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(sbv, value_38, dtype="f32")
                    value_39 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_24, value_38, value_39, kind=BinaryKind.SUB)
                    value_40 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 128),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_39, value_40, dtype="bf16")
                    value_41 = T.tensor_view(
                        T.ptr_of(value_37),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 128),
                    )
                    T.copy(value_40, value_41)
                    for o_m_2 in range(0, 64, 64):
                        for o_n_2 in range(0, 64, 64):
                            for o_k_2 in range(0, 128, 64):
                                value_42 = T.tensor_view(
                                    T.ptr_of(value_29),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                value_43 = T.tensor_view(
                                    T.ptr_of(value_37),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 128),
                                )
                                lhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_43[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_44 = T.tensor_view(
                                    T.ptr_of(value_stages[c % 2]),
                                    layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_44[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_42,
                                    lhs_view_8,
                                    rhs_view_8,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_43[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_44[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_42,
                                    lhs_view_9,
                                    rhs_view_9,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_10 = T.tensor_view(
                                    T.ptr_of(value_43[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_10 = T.tensor_view(
                                    T.ptr_of(value_44[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_42,
                                    lhs_view_10,
                                    rhs_view_10,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_11 = T.tensor_view(
                                    T.ptr_of(value_43[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_11 = T.tensor_view(
                                    T.ptr_of(value_44[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_42,
                                    lhs_view_11,
                                    rhs_view_11,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    dlt = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_45 = T.tensor_view(
                        T.ptr_of(value_29),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.binary(value_28, value_45, dlt, kind=BinaryKind.SUB)
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 64),
                            "f32",
                            ((1, 8, 4 @ scope_3.d2, 2, 1, 1), (0, 512, 128, 64, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_47 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 2]),
                        layout=((1, 64), (0, 1)),
                        shape=(1, 64),
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(value_46),
                        layout=((1, 8, 4 @ scope_3.d2, 2, 1, 1), (0, 512, 128, 64, 0, 0)),
                        shape=(1, 64),
                    )
                    T.copy(value_47, value_48)
                    dk = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(dlt, value_48, dk, kind=BinaryKind.MUL)
                    dkh = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(dk, dkh, dtype="bf16")
                    value_49 = T.tensor_view(
                        T.ptr_of(value_25),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(dkh, value_49)
                    for o_m_3 in range(0, 64, 64):
                        for o_n_3 in range(0, 128, 64):
                            for o_k_3 in range(0, 64, 16):
                                acc_view = T.tensor_view(
                                    T.ptr_of(St_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                value_50 = T.tensor_view(
                                    T.ptr_of(value_25),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_12 = T.tensor_view(
                                    T.ptr_of(value_50[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_51 = T.tensor_view(
                                    T.ptr_of(value_9_stages[c % 2]),
                                    layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 128),
                                )
                                rhs_view_12 = T.tensor_view(
                                    T.ptr_of(value_51[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view,
                                    lhs_view_12,
                                    rhs_view_12,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_52 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(dkh, value_53, dtype="f32")
                    value_54 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(dk, value_53, value_54, kind=BinaryKind.SUB)
                    value_55 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_54, value_55, dtype="bf16")
                    value_56 = T.tensor_view(
                        T.ptr_of(value_52),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_55, value_56)
                    for o_m_4 in range(0, 64, 64):
                        for o_n_4 in range(0, 128, 64):
                            for o_k_4 in range(0, 64, 16):
                                value_57 = T.tensor_view(
                                    T.ptr_of(St_1),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                    shape=(64, 128),
                                )
                                acc_view_1 = T.tensor_view(
                                    T.ptr_of(value_57[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                value_58 = T.tensor_view(
                                    T.ptr_of(value_52),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_13 = T.tensor_view(
                                    T.ptr_of(value_58[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_59 = T.tensor_view(
                                    T.ptr_of(value_9_stages[c % 2]),
                                    layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 128),
                                )
                                rhs_view_13 = T.tensor_view(
                                    T.ptr_of(value_59[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_1,
                                    lhs_view_13,
                                    rhs_view_13,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_61 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(value_61, 0.0)
                    for o_m_5 in range(0, 64, 64):
                        for o_n_5 in range(0, 64, 64):
                            for o_k_5 in range(0, 128, 64):
                                value_62 = T.tensor_view(
                                    T.ptr_of(value_30),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 128),
                                )
                                lhs_view_14 = T.tensor_view(
                                    T.ptr_of(value_62[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_63 = T.tensor_view(
                                    T.ptr_of(value_6_stages[c % 2]),
                                    layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view_14 = T.tensor_view(
                                    T.ptr_of(value_63[o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_61,
                                    lhs_view_14,
                                    rhs_view_14,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_15 = T.tensor_view(
                                    T.ptr_of(value_62[o_m_5:o_m_5 + 64, o_k_5 + 16:o_k_5 + 16 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_15 = T.tensor_view(
                                    T.ptr_of(value_63[o_k_5 + 16:o_k_5 + 16 + 16, o_n_5:o_n_5 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_61,
                                    lhs_view_15,
                                    rhs_view_15,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_16 = T.tensor_view(
                                    T.ptr_of(value_62[o_m_5:o_m_5 + 64, o_k_5 + 32:o_k_5 + 32 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_16 = T.tensor_view(
                                    T.ptr_of(value_63[o_k_5 + 32:o_k_5 + 32 + 16, o_n_5:o_n_5 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_61,
                                    lhs_view_16,
                                    rhs_view_16,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_17 = T.tensor_view(
                                    T.ptr_of(value_62[o_m_5:o_m_5 + 64, o_k_5 + 48:o_k_5 + 48 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_17 = T.tensor_view(
                                    T.ptr_of(value_63[o_k_5 + 48:o_k_5 + 48 + 16, o_n_5:o_n_5 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_61,
                                    lhs_view_17,
                                    rhs_view_17,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_64 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 64),
                            "f32",
                            ((1, 8, 4 @ scope_3.d2, 2, 1, 1), (0, 512, 128, 64, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 2][1:1 + 1, 0:0 + 1]),
                        layout=((1, 64), (0, 1)),
                        shape=(1, 64),
                    )
                    value_66 = T.tensor_view(
                        T.ptr_of(value_64),
                        layout=((1, 8, 4 @ scope_3.d2, 2, 1, 1), (0, 512, 128, 64, 0, 0)),
                        shape=(1, 64),
                    )
                    T.copy(value_65, value_66)
                    value_67 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_68 = T.tensor_view(
                        T.ptr_of(value_61),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.binary(value_68, value_66, value_67, kind=BinaryKind.MUL)
                    value_69 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_70 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(dlt, value_70, dtype="bf16")
                    value_71 = T.tensor_view(
                        T.ptr_of(value_69),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_70, value_71)
                    for o_m_6 in range(0, 64, 64):
                        for o_n_6 in range(0, 64, 64):
                            for o_k_6 in range(0, 64, 64):
                                value_72 = T.tensor_view(
                                    T.ptr_of(value_69),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_18 = T.tensor_view(
                                    T.ptr_of(value_72[o_m_6:o_m_6 + 64, o_k_6:o_k_6 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_73 = T.tensor_view(
                                    T.ptr_of(value_12_stages[c % 2]),
                                    layout=((64, 64), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 64),
                                )
                                rhs_view_18 = T.tensor_view(
                                    T.ptr_of(value_73[o_k_6:o_k_6 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_67,
                                    lhs_view_18,
                                    rhs_view_18,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_19 = T.tensor_view(
                                    T.ptr_of(value_72[o_m_6:o_m_6 + 64, o_k_6 + 16:o_k_6 + 16 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_19 = T.tensor_view(
                                    T.ptr_of(value_73[o_k_6 + 16:o_k_6 + 16 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_67,
                                    lhs_view_19,
                                    rhs_view_19,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_20 = T.tensor_view(
                                    T.ptr_of(value_72[o_m_6:o_m_6 + 64, o_k_6 + 32:o_k_6 + 32 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_20 = T.tensor_view(
                                    T.ptr_of(value_73[o_k_6 + 32:o_k_6 + 32 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_67,
                                    lhs_view_20,
                                    rhs_view_20,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_21 = T.tensor_view(
                                    T.ptr_of(value_72[o_m_6:o_m_6 + 64, o_k_6 + 48:o_k_6 + 48 + 16]),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 2, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                rhs_view_21 = T.tensor_view(
                                    T.ptr_of(value_73[o_k_6 + 48:o_k_6 + 48 + 16, o_n_6:o_n_6 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_67,
                                    lhs_view_21,
                                    rhs_view_21,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_74 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_75 = T.tensor_view(
                        T.ptr_of(value_67),
                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 8, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.cast(value_75, value_74, dtype="bf16")
                    value_76 = T.tensor_view(
                        T.ptr_of(value_60_stages[c % 2]),
                        layout=((64, 64), (1, 72)),
                        shape=(64, 64),
                    )
                    T.copy(value_74, value_76)
                with Mesh(scope[1:], layout=(2, 32, 4), names=("d0", "d1", "d2")) as scope_4:
                    value_78 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((2 @ scope_4.d0, 32 @ scope_4.d1, 4 @ scope_4.d2, 16), (0, 0, 0, 1)),
                            "rmem",
                        ]
                    )
                    value_79 = T.tensor_view(
                        T.ptr_of(value_60_stages[c % 2]),
                        layout=((64, 64), (72, 1)),
                        shape=(64, 64),
                    )
                    value_80 = T.tensor_view(
                        T.ptr_of(value_78),
                        layout=((2 @ scope_4.d0, 32 @ scope_4.d1, 4 @ scope_4.d2, 16), (0, 0, 0, 1)),
                        shape=(64, 64),
                    )
                    T.copy(value_79, value_80)
                    value_81 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((2 @ scope_4.d0, 32 @ scope_4.d1, 4 @ scope_4.d2, 16), (0, 0, 0, 1)),
                            "rmem",
                        ]
                    )
                    T.cast(value_80, value_81, dtype="f32")
                    value_77_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
                    )
                    value_82 = T.tensor_view(
                        T.ptr_of(value_77_1), layout=(64, 64), shape=(64, 64)
                    )
                    T.copy(value_81, value_82)
                    value_83 = T.tensor_view(
                        T.ptr_of(value_77_1),
                        layout=((64, 1, 64), (64, 0, 1)),
                        shape=(64, 1, 64),
                    )
                    value_84 = T.tensor_view(
                        T.ptr_of(out), layout=(256, 48, 128), shape=(256, 48, 128)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_84[c * 64:c * 64 + 64, cta.d0:cta.d0 + 1, cta.d1 * 64:cta.d1 * 64 + 64]),
                        layout=((64, 1, 64), (6144, 128, 1)),
                        shape=(64, 1, 64),
                    )
                    T.copy(value_83, window)
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
                ) as threads:
                    value_85 = T.tensor_view(
                        T.ptr_of(St_1),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 2, 8, 4 @ threads.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        shape=(64, 128),
                    )
                    value_86 = T.tensor_view(
                        T.ptr_of(St),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 2, 8, 4 @ threads.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        shape=(64, 128),
                    )
                    T.copy(value_85, value_86)
            with Mesh(
                (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_5:
                value_88 = T.tensor_view(
                    T.ptr_of(St),
                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 2, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 4096, 512, 128, 64)),
                    shape=(64, 128),
                )
                value_87_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
                )
                value_89 = T.tensor_view(
                    T.ptr_of(value_87_1), layout=((64, 128), (1, 64)), shape=(64, 128)
                )
                T.copy(value_88, value_89)
                value_90 = T.tensor_view(
                    T.ptr_of(value_87_1), layout=((128, 1, 64), (1, 0, 128)), shape=(128, 1, 64)
                )
                value_91 = T.tensor_view(
                    T.ptr_of(out), layout=(256, 48, 128), shape=(256, 48, 128)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_91[128:128 + 128, cta.d0:cta.d0 + 1, cta.d1 * 64:cta.d1 * 64 + 64]),
                    layout=((128, 1, 64), (6144, 128, 1)),
                    shape=(128, 1, 64),
                )
                T.copy(value_90, window_1)
