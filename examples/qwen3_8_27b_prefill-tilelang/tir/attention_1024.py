# analysis target=nvidia.h200_sxm module=Attention function=attention topology=cta wave=132/192
# selection requested=memory executed=memory
# memory traffic=gmem:r1.20MB/w12.12MB@logical,r231.86MB/w2.27GB@total,r1.21MB/w12.12MB@cta,r1.14MB/w12.12MB@thread;rmem:r10.47MB/w8.95MB@logical,r1.96GB/w1.77GB@total,r10.47MB/w9.42MB@cta,r39.67KB/w33.78KB@thread;smem:r1.89MB/w1.14MB@logical,r363.00MB/w219.00MB@total,r1.89MB/w1.14MB@cta,r220.00KB/w1.08MB@thread footprint=iota:512B;v109:88:128.00KB;v111:89:64.00KB footprint-precision=lower_bound peak=gmem:52.06MB;rmem:320.02KB;smem:224.00KB persistent=gmem:28.00MB
#   buffer=iota holds=320B time=kb space=cta.qb,cta.h reuse=535.94KB fits=yes precision=lower_bound
#   error="rmem placement peak 320.02KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attention(
    qkv: Tensor[(1024, 14336), "bf16"], iota: Tensor[(128,), "i32"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(1024, 6144), "bf16"]
):
    with Mesh((Topology("cta", 192),), layout=(8, 24), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        o = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        mx = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        rows = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "i32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "i32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_2 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
        value_4 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
        lsum = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        qs = T.tensor_view(
            0,
            dtype="bf16",
            storage="smem",
            layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
            shape=(128, 256),
        )
        lsum_1 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        corr = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        mnew = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 1), "f32", ((2, 8, 2, 4, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        s0 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        s = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bool",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        dif = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "i32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(128, 80), "i32", (2, 4, 8, 2, 4, 20), "rmem"]
        )
        cols = T.alloc_tensor(
            tensor_type=Tensor[(1, 80), "i32", ((1, 2, 4, 10), (0, 64, 128, 512)), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(1, 80), "i32", ((1, 2, 4, 10), (0, 64, 128, 512)), "rmem"]
        )
        value_14 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
        value_15 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
        value_16 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
        value_17 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 1), "f32", ((2, 8, 2, 4, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        p = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "f32",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", ((2, 8, 2, 4, 1), (64, 1, 8, 16, 0)), "rmem"]
        )
        o_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        pb = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bf16",
                ((2, 5, 8, 2, 4, 2, 4, 2), (5120, 1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 80),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 10), (5120, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        staged = T.tensor_view(
            0,
            dtype="bf16",
            storage="smem",
            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
            shape=(128, 256),
        )
        y = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        ob = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        sg = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        gt = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        tile_9_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(
                scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_1:
                with scope_1 as threads:
                    T.fill(o, 0.0)
                    T.fill(value, 0.0)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1e+30)
                with scope_1 as threads_2:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_2.d0, 8 @ threads_2.d2, 2, 4 @ threads_2.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(mx[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_2.d0, 8 @ threads_2.d2, 2, 4 @ threads_2.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    T.binary(lhs_frame, scalar, dst_frame, kind=BinaryKind.SUB)
                tile = T.tensor_view(T.ptr_of(iota[0:0 + 128]), layout=(128, 1), shape=(128, 1))
                with scope_1 as threads_3:
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_3.d0, 8 @ threads_3.d2, 2, 4 @ threads_3.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    T.copy(tile, dst_frame_1)
                T.fill(value_4, 0.0)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar_1, ((8 - 1) - cta.d0) * 128)
                with scope_1 as threads_4:
                    T.binary(value_4, scalar_1, value_3, kind=BinaryKind.ADD)
                    T.cast(value_3, value_2, dtype="i32")
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_4.d0, 8 @ threads_4.d2, 2, 4 @ threads_4.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(rows[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_4.d0, 8 @ threads_4.d2, 2, 4 @ threads_4.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    T.binary(lhs_frame_1, value_2, dst_frame_2, kind=BinaryKind.ADD)
                    T.fill(lsum, 0.0)
            with scope[2:, :32] as scope_2:
                tile_1 = T.tensor_view(
                    T.ptr_of(qkv[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 128, (cta.d1 * 2) * 256:(cta.d1 * 2) * 256 + 256]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                with Mesh(scope_2, layout=(32,), names=("d0",)) as threads_8:
                    T.copy_async_tensor(tile_1, qs)
            ks_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(106496, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            vs_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            for kb in range(0, ((-2 * cta.d0) + (((32 * cta.d0) - 17) // 80)) + 14, 1):
                with scope[2:, :32] as scope_3:
                    tile_2 = T.tensor_view(
                        T.ptr_of(qkv[kb * 80:kb * 80 + 80, 12288 + ((cta.d1 // 6) * 256):12288 + ((cta.d1 // 6) * 256) + 256]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_9:
                        T.copy_async_tensor(tile_2, ks_stages[kb % 2])
                    tile_3 = T.tensor_view(
                        T.ptr_of(qkv[kb * 80:kb * 80 + 80, 13312 + ((cta.d1 // 6) * 256):13312 + ((cta.d1 // 6) * 256) + 256]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    with Mesh(scope_3, layout=(32,), names=("d0",)) as threads_10:
                        T.copy_async_tensor(tile_3, vs_stages[kb % 2])
                with Mesh(
                    scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    with scope_4 as threads_11:
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(mx[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_11.d0, 8 @ threads_11.d2, 2, 4 @ threads_11.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_11.d0, 8 @ threads_11.d2, 2, 4 @ threads_11.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_2, scalar_2, dst_frame_3, kind=BinaryKind.MUL)
                        T.fill(s0, 0.0)
                    tile_4 = T.tensor_view(
                        T.ptr_of(ks_stages[kb % 2][0:0 + 80, 0:0 + 256]),
                        layout=Layout(((4, 4, 16), (10, 8)), ((5120, 16, 1), (512, 64))) | Swizzle(3, 4, 3),
                        shape=(256, 80),
                    )
                    with Mesh(
                        scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_13:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 80, 80):
                                for o_k in range(0, 256, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(s0[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_13.d1, 2, 4 @ threads_13.d0, 2, 4 @ threads_13.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(qs[o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_4[o_k:o_k + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_13),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(s0[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_13.d1, 2, 4 @ threads_13.d0, 2, 4 @ threads_13.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(qs[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k + 16:o_k + 16 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_13),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(s0[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_13.d1, 2, 4 @ threads_13.d0, 2, 4 @ threads_13.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(qs[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k + 32:o_k + 32 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_13),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(s0[o_m:o_m + 64, o_n:o_n + 80]),
                                        layout=((8 @ threads_13.d1, 2, 4 @ threads_13.d0, 2, 4 @ threads_13.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(qs[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k + 48:o_k + 48 + 16, o_n:o_n + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_13),
                                    )
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_14:
                        for o_m_1 in range(64, 128, 64):
                            for o_n_1 in range(0, 80, 80):
                                for o_k_1 in range(0, 256, 64):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(s0[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(qs[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(s0[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(qs[o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(s0[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(qs[o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(s0[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 80]),
                                        layout=((8 @ threads_14.d1, 2, 4 @ threads_14.d0, 2, 4 @ threads_14.d2, 10), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 80),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(qs[o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_4[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 80]),
                                        layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 80),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_14),
                                    )
                    for _dg in range(((-2 * cta.d0) + (((2 * cta.d0) + 1) // 5)) + 10, kb - (15 * (kb // 15)), 1048576):
                        tile_5 = T.tensor_view(
                            T.ptr_of(iota[0:0 + 80]), layout=(80,), shape=(80,)
                        )
                        tile_6 = T.tensor_view(
                            T.ptr_of(tile_5[0:0 + 80]), layout=(1, 80), shape=(1, 80)
                        )
                        with scope_4 as threads_15:
                            dst_frame_4 = T.tensor_view(
                                T.ptr_of(value_13[0:0 + 1, 0:0 + 80]),
                                layout=((1, 2, 4 @ threads_15.d3, 10), (0, 64, 128, 512)),
                                shape=(1, 80),
                            )
                            T.copy(tile_6, dst_frame_4)
                        T.fill(value_16, 0.0)
                        scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                        T.fill(scalar_3, kb * 80)
                        with scope_4 as threads_16:
                            T.binary(value_16, scalar_3, value_15, kind=BinaryKind.ADD)
                            T.cast(value_15, value_14, dtype="i32")
                            lhs_frame_3 = T.tensor_view(
                                T.ptr_of(value_13[0:0 + 1, 0:0 + 80]),
                                layout=((1, 2, 4 @ threads_16.d3, 10), (0, 64, 128, 512)),
                                shape=(1, 80),
                            )
                            dst_frame_5 = T.tensor_view(
                                T.ptr_of(cols[0:0 + 1, 0:0 + 80]),
                                layout=((1, 2, 4 @ threads_16.d3, 10), (0, 64, 128, 512)),
                                shape=(1, 80),
                            )
                            T.binary(lhs_frame_3, value_14, dst_frame_5, kind=BinaryKind.ADD)
                            lhs_frame_4 = T.tensor_view(
                                T.ptr_of(rows[0:0 + 128, 0:0 + 1]),
                                layout=((2 @ threads_16.d0, 8 @ threads_16.d2, 2, 4 @ threads_16.d1, 1), (64, 1, 8, 16, 0)),
                                shape=(128, 1),
                            )
                            rhs_frame = T.tensor_view(
                                T.ptr_of(cols[0:0 + 1, 0:0 + 80]),
                                layout=((1, 2, 4 @ threads_16.d3, 10), (0, 64, 128, 512)),
                                shape=(1, 80),
                            )
                            dst_frame_6 = T.tensor_view(
                                T.ptr_of(value_12[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_16.d0, 4 @ threads_16.d1, 8 @ threads_16.d2, 2, 4 @ threads_16.d3, 20), (5120, 1280, 160, 80, 20, 1)),
                                shape=(128, 80),
                            )
                            T.binary(lhs_frame_4, rhs_frame, dst_frame_6, kind=BinaryKind.SUB)
                            src_frame = T.tensor_view(
                                T.ptr_of(value_12[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_16.d0, 4 @ threads_16.d1, 8 @ threads_16.d2, 2, 4 @ threads_16.d3, 20), (5120, 1280, 160, 80, 20, 1)),
                                shape=(128, 80),
                            )
                            dst_frame_7 = T.tensor_view(
                                T.ptr_of(dif[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_16.d0, 8 @ threads_16.d2, 2, 4 @ threads_16.d1, 2, 4 @ threads_16.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            T.copy(src_frame, dst_frame_7)
                        T.fill(value_17, 0.0)
                        with scope_4 as threads_21:
                            lhs_frame_5 = T.tensor_view(
                                T.ptr_of(dif[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_21.d0, 8 @ threads_21.d2, 2, 4 @ threads_21.d1, 2, 4 @ threads_21.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            dst_frame_8 = T.tensor_view(
                                T.ptr_of(value_11[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_21.d0, 8 @ threads_21.d2, 2, 4 @ threads_21.d1, 2, 4 @ threads_21.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            T.binary(lhs_frame_5, value_17, dst_frame_8, kind=BinaryKind.GE)
                        scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_4, -1e+30)
                        with scope_4 as threads_22:
                            cond_frame = T.tensor_view(
                                T.ptr_of(value_11[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_22.d0, 8 @ threads_22.d2, 2, 4 @ threads_22.d1, 2, 4 @ threads_22.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            lhs_frame_6 = T.tensor_view(
                                T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_22.d0, 8 @ threads_22.d2, 2, 4 @ threads_22.d1, 2, 4 @ threads_22.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            dst_frame_9 = T.tensor_view(
                                T.ptr_of(s[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_22.d0, 8 @ threads_22.d2, 2, 4 @ threads_22.d1, 2, 4 @ threads_22.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            T.where(cond_frame, lhs_frame_6, scalar_4, dst_frame_9)
                            s_view = T.tensor_view(
                                T.ptr_of(s[0:0 + 128, 0:0 + 80]),
                                layout=((2 @ threads_22.d0, 8 @ threads_22.d2, 2, 4 @ threads_22.d1, 2, 4 @ threads_22.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 80),
                            )
                            T.copy(s_view, s0)
                    with scope_4 as threads_24:
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 2, 4 @ threads_24.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)),
                            shape=(128, 1),
                        )
                        T.reduce(
                            src_frame_1,
                            dst_frame_10,
                            axes=(1,),
                            keepdim=True,
                            kind=ReduceKind.MAX,
                        )
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.copy(src_frame_2, dst_frame_11)
                        lhs_frame_7 = T.tensor_view(
                            T.ptr_of(mx[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(mnew[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_24.d0, 8 @ threads_24.d2, 2, 4 @ threads_24.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_7, rhs_frame_1, dst_frame_12, kind=BinaryKind.MAX)
                    scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_5, 0.09016844005556021)
                    with scope_4 as threads_27:
                        lhs_frame_8 = T.tensor_view(
                            T.ptr_of(mnew[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_8, scalar_5, dst_frame_13, kind=BinaryKind.MUL)
                        lhs_frame_9 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_9, rhs_frame_2, dst_frame_14, kind=BinaryKind.SUB)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(corr[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.unary(src_frame_3, dst_frame_15, kind=UnaryKind.EXP2)
                        lhs_frame_10 = T.tensor_view(
                            T.ptr_of(lsum[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(corr[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_27.d0, 8 @ threads_27.d2, 2, 4 @ threads_27.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_10, rhs_frame_3, dst_frame_16, kind=BinaryKind.MUL)
                    scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_6, 0.09016844005556021)
                    with scope_4 as threads_31:
                        lhs_frame_11 = T.tensor_view(
                            T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_31.d0, 8 @ threads_31.d2, 2, 4 @ threads_31.d1, 2, 4 @ threads_31.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_17 = T.tensor_view(
                            T.ptr_of(value_21[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_31.d0, 8 @ threads_31.d2, 2, 4 @ threads_31.d1, 2, 4 @ threads_31.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        T.binary(lhs_frame_11, scalar_6, dst_frame_17, kind=BinaryKind.MUL)
                    scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_7, 0.09016844005556021)
                    with scope_4 as threads_32:
                        lhs_frame_12 = T.tensor_view(
                            T.ptr_of(mnew[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_18 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_12, scalar_7, dst_frame_18, kind=BinaryKind.MUL)
                        lhs_frame_13 = T.tensor_view(
                            T.ptr_of(value_21[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        rhs_frame_4 = T.tensor_view(
                            T.ptr_of(value_22[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_19 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        T.binary(lhs_frame_13, rhs_frame_4, dst_frame_19, kind=BinaryKind.SUB)
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(value_20[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_20 = T.tensor_view(
                            T.ptr_of(p[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        T.unary(src_frame_4, dst_frame_20, kind=UnaryKind.EXP2)
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(p[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_21 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)),
                            shape=(128, 1),
                        )
                        T.reduce(
                            src_frame_5,
                            dst_frame_21,
                            axes=(1,),
                            keepdim=True,
                            kind=ReduceKind.SUM,
                        )
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(value_19[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1, 1, 1), (64, 8, 4, 1, 0, 0, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_22 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.copy(src_frame_6, dst_frame_22)
                        lhs_frame_14 = T.tensor_view(
                            T.ptr_of(value_5[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        rhs_frame_5 = T.tensor_view(
                            T.ptr_of(value_18[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_23 = T.tensor_view(
                            T.ptr_of(lsum_1[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        T.binary(lhs_frame_14, rhs_frame_5, dst_frame_23, kind=BinaryKind.ADD)
                        lhs_frame_15 = T.tensor_view(
                            T.ptr_of(o[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 256),
                        )
                        rhs_frame_6 = T.tensor_view(
                            T.ptr_of(corr[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 1), (64, 1, 8, 16, 0)),
                            shape=(128, 1),
                        )
                        dst_frame_24 = T.tensor_view(
                            T.ptr_of(o_1[0:0 + 128, 0:0 + 256]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 256),
                        )
                        T.binary(lhs_frame_15, rhs_frame_6, dst_frame_24, kind=BinaryKind.MUL)
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(p[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_25 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        T.cast(src_frame_7, dst_frame_25, dtype="bf16")
                        src_frame_8 = T.tensor_view(
                            T.ptr_of(value_23[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 10), (5120, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        dst_frame_26 = T.tensor_view(
                            T.ptr_of(pb[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ threads_32.d0, 5, 8 @ threads_32.d2, 2, 4 @ threads_32.d1, 2, 4 @ threads_32.d3, 2), (5120, 1024, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 80),
                        )
                        T.copy(src_frame_8, dst_frame_26)
                    tile_7 = T.tensor_view(
                        T.ptr_of(vs_stages[kb % 2][0:0 + 80, 0:0 + 256]),
                        layout=Layout(((5, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    with Mesh(
                        scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_41:
                        for o_m_2 in range(0, 64, 64):
                            for o_n_2 in range(0, 256, 256):
                                for o_k_2 in range(0, 80, 16):
                                    acc_view_8 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 256]),
                                        layout=((8 @ threads_41.d1, 2, 4 @ threads_41.d0, 2, 4 @ threads_41.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_8 = T.tensor_view(
                                        T.ptr_of(pb[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                        layout=((8 @ threads_41.d1, 2, 4 @ threads_41.d0, 2, 4 @ threads_41.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_8 = T.tensor_view(
                                        T.ptr_of(tile_7[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_8,
                                        lhs_view_8,
                                        rhs_view_8,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_41),
                                    )
                    with Mesh(
                        scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")
                    ) as threads_42:
                        for o_m_3 in range(64, 128, 64):
                            for o_n_3 in range(0, 256, 256):
                                for o_k_3 in range(0, 80, 16):
                                    acc_view_9 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 256]),
                                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 32), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 256),
                                    )
                                    lhs_view_9 = T.tensor_view(
                                        T.ptr_of(pb[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                        layout=((8 @ threads_42.d1, 2, 4 @ threads_42.d0, 2, 4 @ threads_42.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_9 = T.tensor_view(
                                        T.ptr_of(tile_7[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 256),
                                    )
                                    T.tiled_mma(
                                        acc_view_9,
                                        lhs_view_9,
                                        rhs_view_9,
                                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_42),
                                    )
                    lsum_1_view = T.tensor_view(
                        T.ptr_of(lsum_1[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.d0, 8 @ scope_4.d2, 2, 4 @ scope_4.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    T.copy(lsum_1_view, lsum)
                    o_1_view = T.tensor_view(
                        T.ptr_of(o_1[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ scope_4.d0, 8 @ scope_4.d2, 2, 4 @ scope_4.d1, 2, 4 @ scope_4.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.copy(o_1_view, o)
                    mnew_view = T.tensor_view(
                        T.ptr_of(mnew[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.d0, 8 @ scope_4.d2, 2, 4 @ scope_4.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    T.copy(mnew_view, mx)
            with Mesh(
                scope[:2], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_5:
                with scope_5 as threads_46:
                    lhs_frame_16 = T.tensor_view(
                        T.ptr_of(o[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    rhs_frame_7 = T.tensor_view(
                        T.ptr_of(lsum[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 1), (64, 1, 8, 16, 0)),
                        shape=(128, 1),
                    )
                    dst_frame_27 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.binary(lhs_frame_16, rhs_frame_7, dst_frame_27, kind=BinaryKind.DIV)
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_28 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_9, dst_frame_28, dtype="bf16")
                    src_frame_10 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_29 = T.tensor_view(
                        T.ptr_of(ob[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_46.d0, 8 @ threads_46.d2, 2, 4 @ threads_46.d1, 2, 4 @ threads_46.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_10, dst_frame_29, dtype="f32")
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 1.0)
                tile_8 = T.tensor_view(
                    T.ptr_of(qkv[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 128, ((cta.d1 * 2) * 256) + 256:((cta.d1 * 2) * 256) + 256 + 256]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                with scope_5 as threads_49:
                    dst_frame_30 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.copy(tile_8, dst_frame_30)
                    src_frame_11 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_31 = T.tensor_view(
                        T.ptr_of(gt[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_11, dst_frame_31, dtype="f32")
                    src_frame_12 = T.tensor_view(
                        T.ptr_of(gt[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_32 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.unary(src_frame_12, dst_frame_32, kind=UnaryKind.NEG)
                    src_frame_13 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_33 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_49.d0, 8 @ threads_49.d2, 2, 4 @ threads_49.d1, 2, 4 @ threads_49.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.unary(src_frame_13, dst_frame_33, kind=UnaryKind.EXP)
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 1.0)
                with scope_5 as threads_53:
                    lhs_frame_17 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_34 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.binary(lhs_frame_17, scalar_9, dst_frame_34, kind=BinaryKind.ADD)
                    rhs_frame_8 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_35 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.binary(scalar_8, rhs_frame_8, dst_frame_35, kind=BinaryKind.DIV)
                    src_frame_14 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_36 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_14, dst_frame_36, dtype="bf16")
                    src_frame_15 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_37 = T.tensor_view(
                        T.ptr_of(sg[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_15, dst_frame_37, dtype="f32")
                    lhs_frame_18 = T.tensor_view(
                        T.ptr_of(ob[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    rhs_frame_9 = T.tensor_view(
                        T.ptr_of(sg[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_38 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.binary(lhs_frame_18, rhs_frame_9, dst_frame_38, kind=BinaryKind.MUL)
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    dst_frame_39 = T.tensor_view(
                        T.ptr_of(y[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.cast(src_frame_16, dst_frame_39, dtype="bf16")
                    src_frame_17 = T.tensor_view(
                        T.ptr_of(y[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ threads_53.d0, 8 @ threads_53.d2, 2, 4 @ threads_53.d1, 2, 4 @ threads_53.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 256),
                    )
                    T.copy(src_frame_17, staged)
            with scope[:1, :32] as scope_6:
                with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_60:
                    T.copy_async_tensor(staged, tile_9_1)
                window = T.tensor_view(
                    T.ptr_of(out[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 128, cta.d1 * 256:cta.d1 * 256 + 256]),
                    layout=((128, 256), (6144, 1)),
                    shape=(128, 256),
                )
                T.copy(tile_9_1, window)
