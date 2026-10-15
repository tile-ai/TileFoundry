# analysis target=nvidia.h200_sxm module=SCHED_GDN_SCAN_T512 function=gdn_scan topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r870.00KB/w15.31MB@logical,r81.56MB/w1.44GB@total,r870.00KB/w15.31MB@cta,r870.00KB/w15.16MB@thread;rmem:r5.23MB/w3.82MB@logical,r501.61MB/w387.38MB@total,r5.23MB/w4.04MB@cta,r32.05KB/w24.77KB@thread;smem:r1.00MB/w774.00KB@logical,r96.38MB/w72.56MB@total,r1.00MB/w774.00KB@cta,r13.28KB/w710.50KB@thread footprint=fc:768B;k:16.00KB;pb:8.00KB;q:16.00KB;ut:16.00KB;v105:114:16.00KB;v113:117:32.00KB footprint-precision=lower_bound peak=gmem:61.31MB;rmem:112.02KB;smem:193.50KB persistent=gmem:31.28MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_scan(
    q: Tensor[(512, 16, 128), "bf16"], k: Tensor[(512, 16, 128), "bf16"], ut: Tensor[(48, 8, 128, 64), "f32"], wb: Tensor[(48, 8, 2, 64, 128), "bf16"], pb: Tensor[(48, 8, 64, 64), "bf16"], fc: Tensor[(48, 8, 3, 64), "f32"], scratch: Tensor[(4096,), "f32"], scratch_1: Tensor[(8192,), "f32"], out: Tensor[(640, 48, 128), "f32"]
):
    with Mesh((Topology("cta", 96),), layout=(48, 2), names=("d0", "d1")) as cta:
        St = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        T.fill(out, 0.0)
        St_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_7 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        dkh = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        dk = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        dlt = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "bf16",
                ((8, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        sbv = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "bf16",
                ((8, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 128),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[
                (1, 64), "f32", ((1, 1, 1, 2, 4, 8), (0, 0, 0, 64, 128, 512)), "rmem"
            ]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[
                (1, 64), "f32", ((1, 1, 1, 2, 4, 8), (0, 0, 0, 64, 128, 512)), "rmem"
            ]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                ((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(64, 64), "f32", ((2, 32, 4, 16), (0, 0, 0, 1)), "rmem"]
        )
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[(64, 64), "bf16", ((2, 32, 4, 16), (0, 0, 0, 1)), "rmem"]
        )
        value_31_1 = T.tensor_view(
            T.ptr_of(scratch_1[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                T.fill(St, 0.0)
            value_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_1_stages = (
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(49152, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_2_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_3_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_4_stages = (
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(172032, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_5_stages = (
                T.tensor_view(131072, dtype="f32", storage="smem", layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(147456, dtype="f32", storage="smem", layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_6_stages = (
                T.tensor_view(196608, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)),
                T.tensor_view(197376, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)),
            )
            value_21_stages = (
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=((64, 64), (1, 72)), shape=(64, 64)),
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=((64, 64), (1, 72)), shape=(64, 64)),
            )
            value_28_1 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
            )
            for c in range(0, 8, 1):
                with scope[1:2, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(wb[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 1, 0:0 + 64, 0:0 + 128]),
                        layout=((1, 1, 1, 64, 128), (131072, 16384, 8192, 128, 1)),
                        shape=(1, 1, 1, 64, 128),
                    )
                    tile_1 = T.tensor_view(
                        T.ptr_of(tile[0:0 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 128]),
                        layout=(64, 128),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile_1, value_stages[c % 2])
                    tile_2 = T.tensor_view(
                        T.ptr_of(wb[cta.d0:cta.d0 + 1, c:c + 1, 1:1 + 1, 0:0 + 64, 0:0 + 128]),
                        layout=((1, 1, 1, 64, 128), (131072, 16384, 8192, 128, 1)),
                        shape=(1, 1, 1, 64, 128),
                    )
                    tile_3 = T.tensor_view(
                        T.ptr_of(tile_2[0:0 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 128]),
                        layout=(64, 128),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_3, value_1_stages[c % 2])
                    tile_4 = T.tensor_view(
                        T.ptr_of(q[c * 64:c * 64 + 64, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 128]),
                        layout=((64, 1, 128), (2048, 128, 1)),
                        shape=(64, 1, 128),
                    )
                    tile_5 = T.tensor_view(
                        T.ptr_of(tile_4[0:0 + 64, 0:0 + 1, 0:0 + 128]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_3:
                        T.copy_async_tensor(tile_5, value_2_stages[c % 2])
                    tile_6 = T.tensor_view(
                        T.ptr_of(k[c * 64:c * 64 + 64, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 128]),
                        layout=((64, 1, 128), (2048, 128, 1)),
                        shape=(64, 1, 128),
                    )
                    tile_7 = T.tensor_view(
                        T.ptr_of(tile_6[0:0 + 64, 0:0 + 1, 0:0 + 128]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_4:
                        T.copy_async_tensor(tile_7, value_3_stages[c % 2])
                    tile_8 = T.tensor_view(
                        T.ptr_of(pb[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 64, 0:0 + 64]),
                        layout=((1, 1, 64, 64), (32768, 4096, 64, 1)),
                        shape=(1, 1, 64, 64),
                    )
                    tile_9 = T.tensor_view(
                        T.ptr_of(tile_8[0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 64]),
                        layout=(64, 64),
                        shape=(64, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_5:
                        T.copy_async_tensor(tile_9, value_4_stages[c % 2])
                    tile_10 = T.tensor_view(
                        T.ptr_of(ut[cta.d0:cta.d0 + 1, c:c + 1, cta.d1 * 64:cta.d1 * 64 + 64, 0:0 + 64]),
                        layout=((1, 1, 64, 64), (65536, 8192, 64, 1)),
                        shape=(1, 1, 64, 64),
                    )
                    tile_11 = T.tensor_view(
                        T.ptr_of(tile_10[0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 64]),
                        layout=(64, 64),
                        shape=(64, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_6:
                        T.copy_async_tensor(tile_11, value_5_stages[c % 2])
                    tile_12 = T.tensor_view(
                        T.ptr_of(fc[cta.d0:cta.d0 + 1, c:c + 1, 0:0 + 3, 0:0 + 64]),
                        layout=((1, 1, 3, 64), (1536, 192, 64, 1)),
                        shape=(1, 1, 3, 64),
                    )
                    tile_13 = T.tensor_view(
                        T.ptr_of(tile_12[0:0 + 1, 0:0 + 1, 0:0 + 3, 0:0 + 64]),
                        layout=(3, 64),
                        shape=(3, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_7:
                        T.copy_async_tensor(tile_13, value_6_stages[c % 2])
                with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_2:
                    tile_14 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2][2:2 + 1, 0:0 + 1]),
                        layout=((1, 1), (64, 1)) + 128,
                        shape=(1, 1),
                    )
                    with scope_2 as threads_8:
                        T.copy(tile_14, value_7)
                        lhs_frame = T.tensor_view(
                            T.ptr_of(St[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame = T.tensor_view(
                            T.ptr_of(St_1[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.binary(lhs_frame, value_7, dst_frame, kind=BinaryKind.MUL)
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(value_5_stages[c % 2], dst_frame_1)
                        T.fill(value_10, 0.0)
                        src_frame = T.tensor_view(
                            T.ptr_of(St[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(sbv[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.cast(src_frame, dst_frame_2, dtype="bf16")
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(sbv[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(value_11[0:0 + 64, 0:0 + 128]),
                            layout=((8, 8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.copy(src_frame_1, dst_frame_3)
                    tile_15 = T.tensor_view(
                        T.ptr_of(value_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_2 as threads_14:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 64, 64):
                                for o_k in range(0, 128, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(value_10[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_11[o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_15[o_k:o_k + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(value_10[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_11[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_15[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(value_10[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_11[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_15[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(value_10[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_11[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_15[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                    tile_16 = T.tensor_view(
                        T.ptr_of(value_1_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_2 as threads_15:
                        for o_m_1 in range(0, 64, 64):
                            for o_n_1 in range(0, 64, 64):
                                for o_k_1 in range(0, 128, 64):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_16[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_16[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_16[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=((8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_16[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_15),
                                    )
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(sbv[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.cast(src_frame_2, dst_frame_4, dtype="f32")
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(St[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_15[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.binary(lhs_frame_1, rhs_frame, dst_frame_5, kind=BinaryKind.SUB)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(value_14[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_6 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.cast(src_frame_3, dst_frame_6, dtype="bf16")
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(value_13[0:0 + 64, 0:0 + 128]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        dst_frame_7 = T.tensor_view(
                            T.ptr_of(value_12[0:0 + 64, 0:0 + 128]),
                            layout=((8, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 128),
                        )
                        T.copy(src_frame_4, dst_frame_7)
                    tile_17 = T.tensor_view(
                        T.ptr_of(value_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_2 as threads_20:
                        for o_m_2 in range(0, 64, 64):
                            for o_n_2 in range(0, 64, 64):
                                for o_k_2 in range(0, 128, 64):
                                    acc_view_8 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_8 = T.tensor_view(
                                        T.ptr_of(value_12[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_8 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_8,
                                        lhs_view_8,
                                        rhs_view_8,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_20),
                                    )
                                    acc_view_9 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_9 = T.tensor_view(
                                        T.ptr_of(value_12[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_9 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_9,
                                        lhs_view_9,
                                        rhs_view_9,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_20),
                                    )
                                    acc_view_10 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_10 = T.tensor_view(
                                        T.ptr_of(value_12[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_10 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_10,
                                        lhs_view_10,
                                        rhs_view_10,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_20),
                                    )
                                    acc_view_11 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_11 = T.tensor_view(
                                        T.ptr_of(value_12[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                        layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_11 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_11,
                                        lhs_view_11,
                                        rhs_view_11,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_20),
                                    )
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_8 = T.tensor_view(
                            T.ptr_of(dlt[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_2, rhs_frame_1, dst_frame_8, kind=BinaryKind.SUB)
                    tile_18 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2][0:0 + 1, 0:0 + 64]),
                        layout=(1, 64),
                        shape=(1, 64),
                    )
                    with scope_2 as threads_22:
                        dst_frame_9 = T.tensor_view(
                            T.ptr_of(value_16[0:0 + 1, 0:0 + 64]),
                            layout=((1, 1, 1, 2, 4 @ threads_22.d2, 8), (0, 0, 0, 64, 128, 512)),
                            shape=(1, 64),
                        )
                        T.copy(tile_18, dst_frame_9)
                        lhs_frame_3 = T.tensor_view(
                            T.ptr_of(dlt[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(value_16[0:0 + 1, 0:0 + 64]),
                            layout=((1, 1, 1, 2, 4 @ threads_22.d2, 8), (0, 0, 0, 64, 128, 512)),
                            shape=(1, 64),
                        )
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(dk[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_3, rhs_frame_2, dst_frame_10, kind=BinaryKind.MUL)
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(dk[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(dkh[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_5, dst_frame_11, dtype="bf16")
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(dkh[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_22.d1, 2, 4 @ threads_22.d0, 2, 4 @ threads_22.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_6, dst_frame_12)
                    tile_19 = T.tensor_view(
                        T.ptr_of(value_3_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    with scope_2 as threads_26:
                        for o_m_3 in range(0, 64, 64):
                            for o_n_3 in range(0, 128, 64):
                                for o_k_3 in range(0, 64, 16):
                                    acc_view_12 = T.tensor_view(
                                        T.ptr_of(St_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                        layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_12 = T.tensor_view(
                                        T.ptr_of(value_8[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                        layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_12 = T.tensor_view(
                                        T.ptr_of(tile_19[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_12,
                                        lhs_view_12,
                                        rhs_view_12,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_26),
                                    )
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(dkh[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_7, dst_frame_13, dtype="f32")
                        lhs_frame_4 = T.tensor_view(
                            T.ptr_of(dk[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_4, rhs_frame_3, dst_frame_14, kind=BinaryKind.SUB)
                        src_frame_8 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_8, dst_frame_15, dtype="bf16")
                        src_frame_9 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(value_17[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_26.d1, 2, 4 @ threads_26.d0, 2, 4 @ threads_26.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_9, dst_frame_16)
                    tile_20 = T.tensor_view(
                        T.ptr_of(value_3_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    with scope_2 as threads_31:
                        for o_m_4 in range(0, 64, 64):
                            for o_n_4 in range(0, 128, 64):
                                for o_k_4 in range(0, 64, 16):
                                    acc_view_13 = T.tensor_view(
                                        T.ptr_of(St_1[o_m_4:o_m_4 + 64, o_n_4:o_n_4 + 64]),
                                        layout=((8 @ threads_31.d1, 2, 4 @ threads_31.d0, 2, 4 @ threads_31.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_13 = T.tensor_view(
                                        T.ptr_of(value_17[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                        layout=((8 @ threads_31.d1, 2, 4 @ threads_31.d0, 2, 4 @ threads_31.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_13 = T.tensor_view(
                                        T.ptr_of(tile_20[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_13,
                                        lhs_view_13,
                                        rhs_view_13,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_31),
                                    )
                        T.fill(value_24, 0.0)
                    tile_21 = T.tensor_view(
                        T.ptr_of(value_2_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_2 as threads_33:
                        for o_m_5 in range(0, 64, 64):
                            for o_n_5 in range(0, 64, 64):
                                for o_k_5 in range(0, 128, 64):
                                    acc_view_14 = T.tensor_view(
                                        T.ptr_of(value_24[o_m_5:o_m_5 + 64, o_n_5:o_n_5 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_14 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_14 = T.tensor_view(
                                        T.ptr_of(tile_21[o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_14,
                                        lhs_view_14,
                                        rhs_view_14,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                    )
                                    acc_view_15 = T.tensor_view(
                                        T.ptr_of(value_24[o_m_5:o_m_5 + 64, o_n_5:o_n_5 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_15 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_5:o_m_5 + 64, o_k_5 + 16:o_k_5 + 16 + 16]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_15 = T.tensor_view(
                                        T.ptr_of(tile_21[o_k_5 + 16:o_k_5 + 16 + 16, o_n_5:o_n_5 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_15,
                                        lhs_view_15,
                                        rhs_view_15,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                    )
                                    acc_view_16 = T.tensor_view(
                                        T.ptr_of(value_24[o_m_5:o_m_5 + 64, o_n_5:o_n_5 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_16 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_5:o_m_5 + 64, o_k_5 + 32:o_k_5 + 32 + 16]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_16 = T.tensor_view(
                                        T.ptr_of(tile_21[o_k_5 + 32:o_k_5 + 32 + 16, o_n_5:o_n_5 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_16,
                                        lhs_view_16,
                                        rhs_view_16,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                    )
                                    acc_view_17 = T.tensor_view(
                                        T.ptr_of(value_24[o_m_5:o_m_5 + 64, o_n_5:o_n_5 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_17 = T.tensor_view(
                                        T.ptr_of(value_11[o_m_5:o_m_5 + 64, o_k_5 + 48:o_k_5 + 48 + 16]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_17 = T.tensor_view(
                                        T.ptr_of(tile_21[o_k_5 + 48:o_k_5 + 48 + 16, o_n_5:o_n_5 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_17,
                                        lhs_view_17,
                                        rhs_view_17,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_33),
                                    )
                    tile_22 = T.tensor_view(
                        T.ptr_of(value_6_stages[c % 2][1:1 + 1, 0:0 + 64]),
                        layout=(1, 64) + 64,
                        shape=(1, 64),
                    )
                    with scope_2 as threads_34:
                        dst_frame_17 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 1, 0:0 + 64]),
                            layout=((1, 1, 1, 2, 4 @ threads_34.d2, 8), (0, 0, 0, 64, 128, 512)),
                            shape=(1, 64),
                        )
                        T.copy(tile_22, dst_frame_17)
                        lhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_24[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_4 = T.tensor_view(
                            T.ptr_of(value_25[0:0 + 1, 0:0 + 64]),
                            layout=((1, 1, 1, 2, 4 @ threads_34.d2, 8), (0, 0, 0, 64, 128, 512)),
                            shape=(1, 64),
                        )
                        dst_frame_18 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_5, rhs_frame_4, dst_frame_18, kind=BinaryKind.MUL)
                        src_frame_10 = T.tensor_view(
                            T.ptr_of(dlt[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_19 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_10, dst_frame_19, dtype="bf16")
                        src_frame_11 = T.tensor_view(
                            T.ptr_of(value_27[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_20 = T.tensor_view(
                            T.ptr_of(value_26[0:0 + 64, 0:0 + 64]),
                            layout=((4, 8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_11, dst_frame_20)
                    tile_23 = T.tensor_view(
                        T.ptr_of(value_4_stages[c % 2][0:0 + 64, 0:0 + 64]),
                        layout=Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    with scope_2 as threads_38:
                        for o_m_6 in range(0, 64, 64):
                            for o_n_6 in range(0, 64, 64):
                                for o_k_6 in range(0, 64, 64):
                                    acc_view_18 = T.tensor_view(
                                        T.ptr_of(value_23[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_18 = T.tensor_view(
                                        T.ptr_of(value_26[o_m_6:o_m_6 + 64, o_k_6:o_k_6 + 16]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_18 = T.tensor_view(
                                        T.ptr_of(tile_23[o_k_6:o_k_6 + 16, o_n_6:o_n_6 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_18,
                                        lhs_view_18,
                                        rhs_view_18,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_38),
                                    )
                                    acc_view_19 = T.tensor_view(
                                        T.ptr_of(value_23[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_19 = T.tensor_view(
                                        T.ptr_of(value_26[o_m_6:o_m_6 + 64, o_k_6 + 16:o_k_6 + 16 + 16]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_19 = T.tensor_view(
                                        T.ptr_of(tile_23[o_k_6 + 16:o_k_6 + 16 + 16, o_n_6:o_n_6 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_19,
                                        lhs_view_19,
                                        rhs_view_19,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_38),
                                    )
                                    acc_view_20 = T.tensor_view(
                                        T.ptr_of(value_23[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_20 = T.tensor_view(
                                        T.ptr_of(value_26[o_m_6:o_m_6 + 64, o_k_6 + 32:o_k_6 + 32 + 16]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_20 = T.tensor_view(
                                        T.ptr_of(tile_23[o_k_6 + 32:o_k_6 + 32 + 16, o_n_6:o_n_6 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_20,
                                        lhs_view_20,
                                        rhs_view_20,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_38),
                                    )
                                    acc_view_21 = T.tensor_view(
                                        T.ptr_of(value_23[o_m_6:o_m_6 + 64, o_n_6:o_n_6 + 64]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_21 = T.tensor_view(
                                        T.ptr_of(value_26[o_m_6:o_m_6 + 64, o_k_6 + 48:o_k_6 + 48 + 16]),
                                        layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_21 = T.tensor_view(
                                        T.ptr_of(tile_23[o_k_6 + 48:o_k_6 + 48 + 16, o_n_6:o_n_6 + 64]),
                                        layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_21,
                                        lhs_view_21,
                                        rhs_view_21,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_38),
                                    )
                        src_frame_12 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_21 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_12, dst_frame_21, dtype="bf16")
                        src_frame_13 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_38.d1, 2, 4 @ threads_38.d0, 2, 4 @ threads_38.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_13, value_21_stages[c % 2])
                with Mesh(scope[1:], layout=(2, 32, 4), names=("d0", "d1", "d2")) as scope_3:
                    tile_24 = T.tensor_view(
                        T.ptr_of(value_21_stages[c % 2][0:0 + 64, 0:0 + 64]),
                        layout=((64, 64), (72, 1)),
                        shape=(64, 64),
                    )
                    with scope_3 as threads_41:
                        dst_frame_22 = T.tensor_view(
                            T.ptr_of(value_30[0:0 + 64, 0:0 + 64]),
                            layout=((2 @ threads_41.d0, 32 @ threads_41.d1, 4 @ threads_41.d2, 16), (0, 0, 0, 1)),
                            shape=(64, 64),
                        )
                        T.copy(tile_24, dst_frame_22)
                        src_frame_14 = T.tensor_view(
                            T.ptr_of(value_30[0:0 + 64, 0:0 + 64]),
                            layout=((2 @ threads_41.d0, 32 @ threads_41.d1, 4 @ threads_41.d2, 16), (0, 0, 0, 1)),
                            shape=(64, 64),
                        )
                        dst_frame_23 = T.tensor_view(
                            T.ptr_of(value_29[0:0 + 64, 0:0 + 64]),
                            layout=((2 @ threads_41.d0, 32 @ threads_41.d1, 4 @ threads_41.d2, 16), (0, 0, 0, 1)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_14, dst_frame_23, dtype="f32")
                        src_frame_15 = T.tensor_view(
                            T.ptr_of(value_29[0:0 + 64, 0:0 + 64]),
                            layout=((2 @ threads_41.d0, 32 @ threads_41.d1, 4 @ threads_41.d2, 16), (0, 0, 0, 1)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_15, value_28_1)
                    tile_25 = T.tensor_view(
                        T.ptr_of(value_28_1[0:0 + 64, 0:0 + 64]),
                        layout=(64, 1, 64),
                        shape=(64, 1, 64),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[c * 64:c * 64 + 64, cta.d0:cta.d0 + 1, cta.d1 * 64:cta.d1 * 64 + 64]),
                        layout=((64, 1, 64), (6144, 128, 1)),
                        shape=(64, 1, 64),
                    )
                    T.copy(tile_25, window)
                with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads_44:
                    St_1_view = T.tensor_view(
                        T.ptr_of(St_1[0:0 + 64, 0:0 + 128]),
                        layout=((2, 8 @ threads_44.d1, 2, 4 @ threads_44.d0, 2, 4 @ threads_44.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 128),
                    )
                    T.copy(St_1_view, St)
            with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_4:
                with scope_4 as threads_45:
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(St[0:0 + 64, 0:0 + 128]),
                        layout=((2, 8 @ threads_45.d1, 2, 4 @ threads_45.d0, 2, 4 @ threads_45.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 128),
                    )
                    T.copy(src_frame_16, value_31_1)
                tile_26 = T.tensor_view(
                    T.ptr_of(value_31_1[0:0 + 64, 0:0 + 128]), layout=(128, 64), shape=(128, 64)
                )
                tile_27 = T.tensor_view(
                    T.ptr_of(tile_26[0:0 + 128, 0:0 + 64]),
                    layout=(128, 1, 64),
                    shape=(128, 1, 64),
                )
                window_1 = T.tensor_view(
                    T.ptr_of(out[512:512 + 128, cta.d0:cta.d0 + 1, cta.d1 * 64:cta.d1 * 64 + 64]),
                    layout=((128, 1, 64), (6144, 128, 1)),
                    shape=(128, 1, 64),
                )
                T.copy(tile_27, window_1)
