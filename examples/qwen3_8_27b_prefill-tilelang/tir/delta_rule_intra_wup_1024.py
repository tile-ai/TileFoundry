# analysis target=nvidia.h200_sxm module=DeltaRuleIntra function=intra_wup topology=cta wave=132/768
# selection requested=memory executed=memory
# memory traffic=gmem:r89.38KB/w30.08MB@logical,r67.03MB/w22.56GB@total,r89.38KB/w30.08MB@cta,r88.15KB/w30.08MB@thread;rmem:r1.73MB/w1.62MB@logical,r1.30GB/w1.21GB@total,r1.73MB/w1.62MB@cta,r31.79KB/w22.74KB@thread;smem:r282.25KB/w228.12KB@logical,r211.69MB/w171.09MB@total,r282.25KB/w228.12KB@cta,r71.34KB/w116.73KB@thread footprint=beta:16.50KB;gc:33.00KB;iota:256B;k:704.00KB;q:704.00KB;v:2.06MB;v417:123:8.00KB;v422:124:1.03MB;v424:257:16.00KB;v427:258:2.06MB;v429:269:16.00KB;v433:270:2.06MB footprint-precision=exact peak=gmem:50.30MB;rmem:260.02KB;smem:73.00KB persistent=gmem:20.28MB
#   buffer=iota holds=8.68MB time=none space=cta.c,cta.h reuse=32.75KB fits=yes precision=exact
#   error="rmem placement peak 260.02KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def intra_wup(
    q: Tensor[(1024, 16, 128), "bf16"], k: Tensor[(1024, 16, 128), "bf16"], v: Tensor[(1024, 48, 128), "bf16"], gc: Tensor[(48, 1024), "f32"], beta: Tensor[(1024, 48), "bf16"], iota: Tensor[(64,), "i32"], scratch: Tensor[(4096,), "bf16"], scratch_1: Tensor[(8192,), "bf16"], scratch_2: Tensor[(8192,), "bf16"], out: Tensor[(1024, 15360), "bf16"]
):
    with Mesh((Topology("cta", 768),), layout=(16, 48), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")) as scope:
            ost = T.tensor_view(
                0,
                dtype="bf16",
                storage="smem",
                layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                shape=(64, 128),
            )
            T.fill(ost, 0.0)
            with Mesh((Topology("thread", 128),), layout=(64,), names=("d0",)) as scope_1:
                gb = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                T.fill(gb, 0.0)
                value = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", (1, 64 @ scope_1.d0), "rmem"]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(gc[cta.d1:cta.d1 + 1, cta.d0 * 64:cta.d0 * 64 + 1]),
                    layout=((1, 64), (0, 1)),
                    shape=(1, 64),
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value), layout=(1, 64 @ scope_1.d0), shape=(1, 64)
                )
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(gb), layout=(4, 64), shape=(4, 64))
                window = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 64]), layout=(1, 64), shape=(1, 64)
                )
                T.copy(value_2, window)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "bf16", (1, 64 @ scope_1.d0), "rmem"]
                )
                value_5 = T.tensor_view(
                    T.ptr_of(beta[cta.d0 * 64:cta.d0 * 64 + 1, cta.d1:cta.d1 + 1]),
                    layout=((1, 64), (0, 48)),
                    shape=(1, 64),
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_4), layout=(1, 64 @ scope_1.d0), shape=(1, 64)
                )
                T.copy(value_5, value_6)
                b1 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", (1, 64 @ scope_1.d0), "rmem"]
                )
                T.cast(value_6, b1, dtype="f32")
                value_7 = T.tensor_view(T.ptr_of(gb), layout=(4, 64), shape=(4, 64))
                window_1 = T.tensor_view(
                    T.ptr_of(value_7[1:1 + 1, 0:0 + 64]), layout=(1, 64) + 64, shape=(1, 64)
                )
                T.copy(b1, window_1)
            with scope[:1] as scope_2:
                value_9 = T.tensor_view(
                    T.ptr_of(k[cta.d0 * 64:cta.d0 * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                    layout=((64, 128), (2048, 1)),
                    shape=(64, 128),
                )
                value_8 = T.tensor_view(
                    17408,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_10 = T.tensor_view(
                    T.ptr_of(value_8),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_9, value_10)
                value_12 = T.tensor_view(
                    T.ptr_of(q[cta.d0 * 64:cta.d0 * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                    layout=((64, 128), (2048, 1)),
                    shape=(64, 128),
                )
                value_11 = T.tensor_view(
                    33792,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_13 = T.tensor_view(
                    T.ptr_of(value_11),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_12, value_13)
            value_14 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 8, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_17 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 8, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_3:
                T.fill(value_14, 0.0)
                for o_m in range(0, 64, 64):
                    for o_n in range(0, 64, 64):
                        for o_k in range(0, 128, 64):
                            value_8_2 = T.tensor_view(
                                17408,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_15 = T.tensor_view(
                                T.ptr_of(value_8_2),
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            lhs_view = T.tensor_view(
                                T.ptr_of(value_15[o_m:o_m + 64, o_k:o_k + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_16 = T.tensor_view(
                                T.ptr_of(value_8_2),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            rhs_view = T.tensor_view(
                                T.ptr_of(value_16[o_k:o_k + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_14,
                                lhs_view,
                                rhs_view,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_1 = T.tensor_view(
                                T.ptr_of(value_15[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_1 = T.tensor_view(
                                T.ptr_of(value_16[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_14,
                                lhs_view_1,
                                rhs_view_1,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_2 = T.tensor_view(
                                T.ptr_of(value_15[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_2 = T.tensor_view(
                                T.ptr_of(value_16[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_14,
                                lhs_view_2,
                                rhs_view_2,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_3 = T.tensor_view(
                                T.ptr_of(value_15[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_3 = T.tensor_view(
                                T.ptr_of(value_16[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_14,
                                lhs_view_3,
                                rhs_view_3,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                T.fill(value_17, 0.0)
                for o_m_1 in range(0, 64, 64):
                    for o_n_1 in range(0, 64, 64):
                        for o_k_1 in range(0, 128, 64):
                            value_11_2 = T.tensor_view(
                                33792,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_18 = T.tensor_view(
                                T.ptr_of(value_11_2),
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            lhs_view_4 = T.tensor_view(
                                T.ptr_of(value_18[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_8_3 = T.tensor_view(
                                17408,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_19 = T.tensor_view(
                                T.ptr_of(value_8_3),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            rhs_view_4 = T.tensor_view(
                                T.ptr_of(value_19[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_17,
                                lhs_view_4,
                                rhs_view_4,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_5 = T.tensor_view(
                                T.ptr_of(value_18[o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_5 = T.tensor_view(
                                T.ptr_of(value_19[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_17,
                                lhs_view_5,
                                rhs_view_5,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_6 = T.tensor_view(
                                T.ptr_of(value_18[o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_6 = T.tensor_view(
                                T.ptr_of(value_19[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_17,
                                lhs_view_6,
                                rhs_view_6,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_7 = T.tensor_view(
                                T.ptr_of(value_18[o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_7 = T.tensor_view(
                                T.ptr_of(value_19[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_17,
                                lhs_view_7,
                                rhs_view_7,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_4:
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "i32",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                value_23 = T.tensor_view(
                    T.ptr_of(iota), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_24 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                    shape=(64, 1),
                )
                T.copy(value_23, value_24)
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "i32", ((1, 2, 4 @ scope_4.d2, 8), (0, 64, 128, 512)), "rmem"
                    ]
                )
                value_26 = T.tensor_view(
                    T.ptr_of(iota), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_27 = T.tensor_view(
                    T.ptr_of(value_25),
                    layout=((1, 2, 4 @ scope_4.d2, 8), (0, 64, 128, 512)),
                    shape=(1, 64),
                )
                T.copy(value_26, value_27)
                value_28 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_24, value_27, value_28, kind=BinaryKind.SUB)
                value_29 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_28, value_29)
                zi = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
                T.fill(zi, 0.0)
                value_30 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bool",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_29, zi, value_30, kind=BinaryKind.GE)
                value_31 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                gb_2 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                value_32 = T.tensor_view(
                    T.ptr_of(gb_2), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_33 = T.tensor_view(
                    T.ptr_of(value_31),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                    shape=(64, 1),
                )
                T.copy(value_32, value_33)
                value_34 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "f32", ((1, 2, 4 @ scope_4.d2, 8), (0, 64, 128, 512)), "rmem"
                    ]
                )
                value_35 = T.tensor_view(
                    T.ptr_of(gb_2), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_36 = T.tensor_view(
                    T.ptr_of(value_34),
                    layout=((1, 2, 4 @ scope_4.d2, 8), (0, 64, 128, 512)),
                    shape=(1, 64),
                )
                T.copy(value_35, value_36)
                value_37 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_33, value_36, value_37, kind=BinaryKind.SUB)
                dec = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_37, dec, kind=UnaryKind.EXP)
                value_38 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                value_39 = T.tensor_view(
                    T.ptr_of(value_17),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_39, dec, value_38, kind=BinaryKind.MUL)
                value_40 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 0.08838834764831845)
                T.binary(value_38, scalar, value_40, kind=BinaryKind.MUL)
                value_41 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 0.0)
                T.where(value_30, value_40, scalar_1, value_41)
                pm = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_41, pm, dtype="bf16")
                value_20 = T.tensor_view(
                    33792,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_42 = T.tensor_view(
                    T.ptr_of(value_20),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(pm, value_42)
                asm = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                T.fill(asm, 0.0)
                value_43 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bool",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_29, zi, value_43, kind=BinaryKind.GT)
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                value_45 = T.tensor_view(
                    T.ptr_of(gb_2[1:1 + 1, 0:0 + 1]), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_46 = T.tensor_view(
                    T.ptr_of(value_44),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 1), (16, 8, 1, 0)),
                    shape=(64, 1),
                )
                T.copy(value_45, value_46)
                value_47 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_48 = T.tensor_view(
                    T.ptr_of(value_14),
                    layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_48, value_46, value_47, kind=BinaryKind.MUL)
                value_49 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_47, dec, value_49, kind=BinaryKind.MUL)
                value_50 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_49, value_50, kind=UnaryKind.NEG)
                value_51 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_4.d0, 8 @ scope_4.d1, 2, 4 @ scope_4.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 0.0)
                T.where(value_43, value_50, scalar_2, value_51)
                value_52 = T.tensor_view(T.ptr_of(asm), layout=(64, 65), shape=(64, 65))
                window_2 = T.tensor_view(
                    T.ptr_of(value_52[0:0 + 64, 0:0 + 64]),
                    layout=((64, 64), (65, 1)),
                    shape=(64, 64),
                )
                T.copy(value_51, window_2)
            with Mesh((Topology("thread", 128),), layout=(16,), names=("d0",)) as scope_5:
                xb0 = T.alloc_tensor(
                    tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_5.d0), "rmem"]
                )
                T.fill(xb0, 0.0)
                for i in range(0, 16, 1):
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_5.d0), "rmem"]
                    )
                    asm_2 = T.tensor_view(
                        41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                    )
                    value_54 = T.tensor_view(
                        T.ptr_of(asm_2[0 + i:0 + i + 1, 0:0 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_55 = T.tensor_view(
                        T.ptr_of(value_53), layout=(1, 16 @ scope_5.d0), shape=(1, 16)
                    )
                    T.copy(value_54, value_55)
                    value_56 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_5.d0), "rmem"]
                    )
                    value_57 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_5.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_58 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_59 = T.tensor_view(
                        T.ptr_of(asm_2[0 + i:0 + i + 1, 0:0 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_60 = T.tensor_view(T.ptr_of(value_58), layout=(16, 1), shape=(16, 1))
                    T.copy(value_59, value_60)
                    value_61 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_5.d0), "rmem"]
                    )
                    value_62 = T.tensor_view(
                        T.ptr_of(xb0), layout=(16, 16 @ scope_5.d0), shape=(16, 16)
                    )
                    T.binary(value_60, value_62, value_61, kind=BinaryKind.MUL)
                    value_63 = T.tensor_view(
                        T.ptr_of(value_57),
                        layout=((16, 16 @ scope_5.d0), (1, 0)),
                        shape=(16, 16),
                    )
                    T.copy(value_61, value_63)
                    value_64 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_5.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_63, value_64, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_65 = T.tensor_view(
                        T.ptr_of(value_56), layout=(1, 16 @ scope_5.d0), shape=(1, 16)
                    )
                    T.copy(value_64, value_65)
                    value_66 = T.tensor_view(
                        T.ptr_of(xb0[i:i + 1, 0:0 + 16]),
                        layout=(1, 16 @ scope_5.d0),
                        shape=(1, 16),
                    )
                    T.binary(value_55, value_65, value_66, kind=BinaryKind.ADD)
                value_67 = T.tensor_view(
                    T.ptr_of(xb0), layout=(16, 16 @ scope_5.d0), shape=(16, 16)
                )
                asm_3 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_68 = T.tensor_view(T.ptr_of(asm_3), layout=(64, 65), shape=(64, 65))
                window_3 = T.tensor_view(
                    T.ptr_of(value_68[0:0 + 16, 0:0 + 16]),
                    layout=((16, 16), (65, 1)),
                    shape=(16, 16),
                )
                T.copy(value_67, window_3)
            with Mesh(scope[:1, 16:], layout=(16,), names=("d0",)) as scope_6:
                xb1 = T.alloc_tensor(
                    tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_6.d0), "rmem"]
                )
                T.fill(xb1, 0.0)
                for i_2 in range(0, 16, 1):
                    value_69 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_6.d0), "rmem"]
                    )
                    asm_4 = T.tensor_view(
                        41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                    )
                    value_70 = T.tensor_view(
                        T.ptr_of(asm_4[16 + i_2:16 + i_2 + 1, 16:16 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_71 = T.tensor_view(
                        T.ptr_of(value_69), layout=(1, 16 @ scope_6.d0), shape=(1, 16)
                    )
                    T.copy(value_70, value_71)
                    value_72 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_6.d0), "rmem"]
                    )
                    value_73 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_6.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_74 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_75 = T.tensor_view(
                        T.ptr_of(asm_4[16 + i_2:16 + i_2 + 1, 16:16 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_76 = T.tensor_view(T.ptr_of(value_74), layout=(16, 1), shape=(16, 1))
                    T.copy(value_75, value_76)
                    value_77 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_6.d0), "rmem"]
                    )
                    value_78 = T.tensor_view(
                        T.ptr_of(xb1), layout=(16, 16 @ scope_6.d0), shape=(16, 16)
                    )
                    T.binary(value_76, value_78, value_77, kind=BinaryKind.MUL)
                    value_79 = T.tensor_view(
                        T.ptr_of(value_73),
                        layout=((16, 16 @ scope_6.d0), (1, 0)),
                        shape=(16, 16),
                    )
                    T.copy(value_77, value_79)
                    value_80 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_6.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_79, value_80, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_81 = T.tensor_view(
                        T.ptr_of(value_72), layout=(1, 16 @ scope_6.d0), shape=(1, 16)
                    )
                    T.copy(value_80, value_81)
                    value_82 = T.tensor_view(
                        T.ptr_of(xb1[i_2:i_2 + 1, 0:0 + 16]),
                        layout=(1, 16 @ scope_6.d0),
                        shape=(1, 16),
                    )
                    T.binary(value_71, value_81, value_82, kind=BinaryKind.ADD)
                value_83 = T.tensor_view(
                    T.ptr_of(xb1), layout=(16, 16 @ scope_6.d0), shape=(16, 16)
                )
                asm_5 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_84 = T.tensor_view(T.ptr_of(asm_5), layout=(64, 65), shape=(64, 65))
                window_4 = T.tensor_view(
                    T.ptr_of(value_84[16:16 + 16, 16:16 + 16]),
                    layout=((16, 16), (65, 1)) + 1056,
                    shape=(16, 16),
                )
                T.copy(value_83, window_4)
            with Mesh(scope[1:2, :16], layout=(16,), names=("d0",)) as scope_7:
                xb2 = T.alloc_tensor(
                    tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_7.d0), "rmem"]
                )
                T.fill(xb2, 0.0)
                for i_3 in range(0, 16, 1):
                    value_85 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_7.d0), "rmem"]
                    )
                    asm_6 = T.tensor_view(
                        41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                    )
                    value_86 = T.tensor_view(
                        T.ptr_of(asm_6[32 + i_3:32 + i_3 + 1, 32:32 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_87 = T.tensor_view(
                        T.ptr_of(value_85), layout=(1, 16 @ scope_7.d0), shape=(1, 16)
                    )
                    T.copy(value_86, value_87)
                    value_88 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_7.d0), "rmem"]
                    )
                    value_89 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_7.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_90 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_91 = T.tensor_view(
                        T.ptr_of(asm_6[32 + i_3:32 + i_3 + 1, 32:32 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_92 = T.tensor_view(T.ptr_of(value_90), layout=(16, 1), shape=(16, 1))
                    T.copy(value_91, value_92)
                    value_93 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_7.d0), "rmem"]
                    )
                    value_94 = T.tensor_view(
                        T.ptr_of(xb2), layout=(16, 16 @ scope_7.d0), shape=(16, 16)
                    )
                    T.binary(value_92, value_94, value_93, kind=BinaryKind.MUL)
                    value_95 = T.tensor_view(
                        T.ptr_of(value_89),
                        layout=((16, 16 @ scope_7.d0), (1, 0)),
                        shape=(16, 16),
                    )
                    T.copy(value_93, value_95)
                    value_96 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_7.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_95, value_96, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_97 = T.tensor_view(
                        T.ptr_of(value_88), layout=(1, 16 @ scope_7.d0), shape=(1, 16)
                    )
                    T.copy(value_96, value_97)
                    value_98 = T.tensor_view(
                        T.ptr_of(xb2[i_3:i_3 + 1, 0:0 + 16]),
                        layout=(1, 16 @ scope_7.d0),
                        shape=(1, 16),
                    )
                    T.binary(value_87, value_97, value_98, kind=BinaryKind.ADD)
                value_99 = T.tensor_view(
                    T.ptr_of(xb2), layout=(16, 16 @ scope_7.d0), shape=(16, 16)
                )
                asm_7 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_100 = T.tensor_view(T.ptr_of(asm_7), layout=(64, 65), shape=(64, 65))
                window_5 = T.tensor_view(
                    T.ptr_of(value_100[32:32 + 16, 32:32 + 16]),
                    layout=((16, 16), (65, 1)) + 2112,
                    shape=(16, 16),
                )
                T.copy(value_99, window_5)
            with Mesh(scope[1:2, 16:], layout=(16,), names=("d0",)) as scope_8:
                xb3 = T.alloc_tensor(
                    tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_8.d0), "rmem"]
                )
                T.fill(xb3, 0.0)
                for i_4 in range(0, 16, 1):
                    value_101 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_8.d0), "rmem"]
                    )
                    asm_8 = T.tensor_view(
                        41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                    )
                    value_102 = T.tensor_view(
                        T.ptr_of(asm_8[48 + i_4:48 + i_4 + 1, 48:48 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_103 = T.tensor_view(
                        T.ptr_of(value_101), layout=(1, 16 @ scope_8.d0), shape=(1, 16)
                    )
                    T.copy(value_102, value_103)
                    value_104 = T.alloc_tensor(
                        tensor_type=Tensor[(1, 16), "f32", (1, 16 @ scope_8.d0), "rmem"]
                    )
                    value_105 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_8.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_106 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_107 = T.tensor_view(
                        T.ptr_of(asm_8[48 + i_4:48 + i_4 + 1, 48:48 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_108 = T.tensor_view(
                        T.ptr_of(value_106), layout=(16, 1), shape=(16, 1)
                    )
                    T.copy(value_107, value_108)
                    value_109 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 16), "f32", (16, 16 @ scope_8.d0), "rmem"]
                    )
                    value_110 = T.tensor_view(
                        T.ptr_of(xb3), layout=(16, 16 @ scope_8.d0), shape=(16, 16)
                    )
                    T.binary(value_108, value_110, value_109, kind=BinaryKind.MUL)
                    value_111 = T.tensor_view(
                        T.ptr_of(value_105),
                        layout=((16, 16 @ scope_8.d0), (1, 0)),
                        shape=(16, 16),
                    )
                    T.copy(value_109, value_111)
                    value_112 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_8.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_111, value_112, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_113 = T.tensor_view(
                        T.ptr_of(value_104), layout=(1, 16 @ scope_8.d0), shape=(1, 16)
                    )
                    T.copy(value_112, value_113)
                    value_114 = T.tensor_view(
                        T.ptr_of(xb3[i_4:i_4 + 1, 0:0 + 16]),
                        layout=(1, 16 @ scope_8.d0),
                        shape=(1, 16),
                    )
                    T.binary(value_103, value_113, value_114, kind=BinaryKind.ADD)
                value_115 = T.tensor_view(
                    T.ptr_of(xb3), layout=(16, 16 @ scope_8.d0), shape=(16, 16)
                )
                asm_9 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_116 = T.tensor_view(T.ptr_of(asm_9), layout=(64, 65), shape=(64, 65))
                window_6 = T.tensor_view(
                    T.ptr_of(value_116[48:48 + 16, 48:48 + 16]),
                    layout=((16, 16), (65, 1)) + 3168,
                    shape=(16, 16),
                )
                T.copy(value_115, window_6)
            value_117 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 2, 8, 1), None),
                        attrs=(S(2), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_122 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 1, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 1, 16), None),
                        attrs=(B(), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_125 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 2, 8, 16), None),
                        attrs=(S(2), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_136 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 1, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 1, 16), None),
                        attrs=(B(), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_139 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 2, 8, 16), None),
                        attrs=(S(2), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            t21t = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 2, 8, 1), None),
                        attrs=(S(2), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_9:
                asm_10 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_118 = T.tensor_view(
                    T.ptr_of(asm_10[16:16 + 1, 0:0 + 1]),
                    layout=((16, 16, 1), (1, 65, 0)),
                    shape=(16, 16, 1),
                )
                value_119 = T.tensor_view(
                    T.ptr_of(value_117),
                    layout=(16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_118, value_119)
                value_120 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1), "rmem"
                    ]
                )
                value_121 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 16), (0, 16, 0, 1)),
                        "rmem",
                    ]
                )
                value_123 = T.tensor_view(
                    T.ptr_of(asm_10), layout=((16, 1, 16), (1, 0, 65)), shape=(16, 1, 16)
                )
                value_124 = T.tensor_view(
                    T.ptr_of(value_122), layout=(16 @ scope_9.d1, 1, 16), shape=(16, 1, 16)
                )
                T.copy(value_123, value_124)
                value_126 = T.tensor_view(
                    T.ptr_of(asm_10[16:16 + 1, 0:0 + 1]),
                    layout=((1, 16, 16), (0, 65, 1)),
                    shape=(1, 16, 16),
                )
                value_127 = T.tensor_view(
                    T.ptr_of(value_125), layout=(1, 2, 8 @ scope_9.d0, 16), shape=(1, 16, 16)
                )
                T.copy(value_126, value_127)
                value_128 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_9.d1, 8 @ scope_9.d0, 2, 16), (256, 32, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_124, value_127, value_128, kind=BinaryKind.MUL)
                value_129 = T.tensor_view(
                    T.ptr_of(value_121),
                    layout=((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 16), (0, 16, 0, 1)),
                    shape=(16, 16, 16),
                )
                T.copy(value_128, value_129)
                value_130 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1),
                        "f32",
                        ((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_129, value_130, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_131 = T.tensor_view(
                    T.ptr_of(value_120),
                    layout=(16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_130, value_131)
                xt0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1), "rmem"
                    ]
                )
                T.binary(value_119, value_131, xt0, kind=BinaryKind.ADD)
                value_132 = T.tensor_view(
                    T.ptr_of(xt0), layout=(16 @ scope_9.d1, 2, 8 @ scope_9.d0), shape=(16, 16)
                )
                value_133 = T.tensor_view(T.ptr_of(asm_10), layout=(64, 65), shape=(64, 65))
                window_7 = T.tensor_view(
                    T.ptr_of(value_133[0:0 + 16, 16:16 + 16]),
                    layout=((16, 16), (65, 1)) + 16,
                    shape=(16, 16),
                )
                T.copy(value_132, window_7)
                value_134 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1), "rmem"
                    ]
                )
                value_135 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 16), (0, 16, 0, 1)),
                        "rmem",
                    ]
                )
                value_137 = T.tensor_view(
                    T.ptr_of(asm_10[0:0 + 1, 16:16 + 1]),
                    layout=((16, 1, 16), (65, 0, 1)),
                    shape=(16, 1, 16),
                )
                value_138 = T.tensor_view(
                    T.ptr_of(value_136), layout=(16 @ scope_9.d1, 1, 16), shape=(16, 1, 16)
                )
                T.copy(value_137, value_138)
                value_140 = T.tensor_view(
                    T.ptr_of(asm_10[16:16 + 1, 16:16 + 1]),
                    layout=((1, 16, 16), (0, 65, 1)),
                    shape=(1, 16, 16),
                )
                value_141 = T.tensor_view(
                    T.ptr_of(value_139), layout=(1, 2, 8 @ scope_9.d0, 16), shape=(1, 16, 16)
                )
                T.copy(value_140, value_141)
                value_142 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_9.d1, 8 @ scope_9.d0, 2, 16), (256, 32, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_138, value_141, value_142, kind=BinaryKind.MUL)
                value_143 = T.tensor_view(
                    T.ptr_of(value_135),
                    layout=((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 16), (0, 16, 0, 1)),
                    shape=(16, 16, 16),
                )
                T.copy(value_142, value_143)
                value_144 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1),
                        "f32",
                        ((16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_143, value_144, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_145 = T.tensor_view(
                    T.ptr_of(value_134),
                    layout=(16 @ scope_9.d1, 2, 8 @ scope_9.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_144, value_145)
                T.binary(xt0, value_145, t21t, kind=BinaryKind.ADD)
                value_146 = T.tensor_view(
                    T.ptr_of(t21t), layout=(16 @ scope_9.d1, 2, 8 @ scope_9.d0), shape=(16, 16)
                )
                value_147 = T.tensor_view(T.ptr_of(asm_10), layout=(64, 65), shape=(64, 65))
                window_8 = T.tensor_view(
                    T.ptr_of(value_147[0:0 + 16, 16:16 + 16]),
                    layout=((16, 16), (65, 1)) + 16,
                    shape=(16, 16),
                )
                T.copy(value_146, window_8)
            value_148 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 2, 8, 1), None),
                        attrs=(S(2), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_153 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 1, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 1, 16), None),
                        attrs=(B(), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_156 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 2, 8, 16), None),
                        attrs=(S(2), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_167 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 1, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 1, 16), None),
                        attrs=(B(), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_170 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 2, 8, 16), None),
                        attrs=(S(2), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            t21t_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 2, 8, 1), None),
                        attrs=(S(2), S(0)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_10:
                asm_11 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_149 = T.tensor_view(
                    T.ptr_of(asm_11[48:48 + 1, 32:32 + 1]),
                    layout=((16, 16, 1), (1, 65, 0)),
                    shape=(16, 16, 1),
                )
                value_150 = T.tensor_view(
                    T.ptr_of(value_148),
                    layout=(16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_149, value_150)
                value_151 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1), "rmem"
                    ]
                )
                value_152 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 16), (0, 16, 0, 1)),
                        "rmem",
                    ]
                )
                value_154 = T.tensor_view(
                    T.ptr_of(asm_11[32:32 + 1, 32:32 + 1]),
                    layout=((16, 1, 16), (1, 0, 65)),
                    shape=(16, 1, 16),
                )
                value_155 = T.tensor_view(
                    T.ptr_of(value_153), layout=(16 @ scope_10.d1, 1, 16), shape=(16, 1, 16)
                )
                T.copy(value_154, value_155)
                value_157 = T.tensor_view(
                    T.ptr_of(asm_11[48:48 + 1, 32:32 + 1]),
                    layout=((1, 16, 16), (0, 65, 1)),
                    shape=(1, 16, 16),
                )
                value_158 = T.tensor_view(
                    T.ptr_of(value_156), layout=(1, 2, 8 @ scope_10.d0, 16), shape=(1, 16, 16)
                )
                T.copy(value_157, value_158)
                value_159 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_10.d1, 8 @ scope_10.d0, 2, 16), (256, 32, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_155, value_158, value_159, kind=BinaryKind.MUL)
                value_160 = T.tensor_view(
                    T.ptr_of(value_152),
                    layout=((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 16), (0, 16, 0, 1)),
                    shape=(16, 16, 16),
                )
                T.copy(value_159, value_160)
                value_161 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1),
                        "f32",
                        ((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_160, value_161, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_162 = T.tensor_view(
                    T.ptr_of(value_151),
                    layout=(16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_161, value_162)
                xt1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1), "rmem"
                    ]
                )
                T.binary(value_150, value_162, xt1, kind=BinaryKind.ADD)
                value_163 = T.tensor_view(
                    T.ptr_of(xt1), layout=(16 @ scope_10.d1, 2, 8 @ scope_10.d0), shape=(16, 16)
                )
                value_164 = T.tensor_view(T.ptr_of(asm_11), layout=(64, 65), shape=(64, 65))
                window_9 = T.tensor_view(
                    T.ptr_of(value_164[32:32 + 16, 48:48 + 16]),
                    layout=((16, 16), (65, 1)) + 2128,
                    shape=(16, 16),
                )
                T.copy(value_163, window_9)
                value_165 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1), "f32", (16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1), "rmem"
                    ]
                )
                value_166 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 16), (0, 16, 0, 1)),
                        "rmem",
                    ]
                )
                value_168 = T.tensor_view(
                    T.ptr_of(asm_11[32:32 + 1, 48:48 + 1]),
                    layout=((16, 1, 16), (65, 0, 1)),
                    shape=(16, 1, 16),
                )
                value_169 = T.tensor_view(
                    T.ptr_of(value_167), layout=(16 @ scope_10.d1, 1, 16), shape=(16, 1, 16)
                )
                T.copy(value_168, value_169)
                value_171 = T.tensor_view(
                    T.ptr_of(asm_11[48:48 + 1, 48:48 + 1]),
                    layout=((1, 16, 16), (0, 65, 1)),
                    shape=(1, 16, 16),
                )
                value_172 = T.tensor_view(
                    T.ptr_of(value_170), layout=(1, 2, 8 @ scope_10.d0, 16), shape=(1, 16, 16)
                )
                T.copy(value_171, value_172)
                value_173 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_10.d1, 8 @ scope_10.d0, 2, 16), (256, 32, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_169, value_172, value_173, kind=BinaryKind.MUL)
                value_174 = T.tensor_view(
                    T.ptr_of(value_166),
                    layout=((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 16), (0, 16, 0, 1)),
                    shape=(16, 16, 16),
                )
                T.copy(value_173, value_174)
                value_175 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 1),
                        "f32",
                        ((16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_174, value_175, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_176 = T.tensor_view(
                    T.ptr_of(value_165),
                    layout=(16 @ scope_10.d1, 2, 8 @ scope_10.d0, 1),
                    shape=(16, 16, 1),
                )
                T.copy(value_175, value_176)
                T.binary(xt1, value_176, t21t_1, kind=BinaryKind.ADD)
                value_177 = T.tensor_view(
                    T.ptr_of(t21t_1),
                    layout=(16 @ scope_10.d1, 2, 8 @ scope_10.d0),
                    shape=(16, 16),
                )
                value_178 = T.tensor_view(T.ptr_of(asm_11), layout=(64, 65), shape=(64, 65))
                window_10 = T.tensor_view(
                    T.ptr_of(value_178[32:32 + 16, 48:48 + 16]),
                    layout=((16, 16), (65, 1)) + 2128,
                    shape=(16, 16),
                )
                T.copy(value_177, window_10)
            maa = T.tensor_view(
                58624, dtype="f32", storage="smem", layout=(32, 33), shape=(32, 33)
            )
            T.fill(maa, 0.0)
            asm_12 = T.tensor_view(
                41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
            )
            value_179 = T.tensor_view(
                T.ptr_of(asm_12), layout=((16, 16), (65, 1)), shape=(16, 16)
            )
            value_180 = T.tensor_view(T.ptr_of(maa), layout=(32, 33), shape=(32, 33))
            window_11 = T.tensor_view(
                T.ptr_of(value_180[0:0 + 16, 0:0 + 16]),
                layout=((16, 16), (33, 1)),
                shape=(16, 16),
            )
            T.copy(value_179, window_11)
            value_181 = T.tensor_view(
                T.ptr_of(asm_12[16:16 + 1, 16:16 + 1]),
                layout=((16, 16), (65, 1)),
                shape=(16, 16),
            )
            value_182 = T.tensor_view(T.ptr_of(maa), layout=(32, 33), shape=(32, 33))
            window_12 = T.tensor_view(
                T.ptr_of(value_182[16:16 + 16, 16:16 + 16]),
                layout=((16, 16), (33, 1)) + 544,
                shape=(16, 16),
            )
            T.copy(value_181, window_12)
            value_183 = T.tensor_view(
                T.ptr_of(asm_12[0:0 + 1, 16:16 + 1]), layout=((16, 16), (1, 65)), shape=(16, 16)
            )
            value_184 = T.tensor_view(T.ptr_of(maa), layout=(32, 33), shape=(32, 33))
            window_13 = T.tensor_view(
                T.ptr_of(value_184[16:16 + 16, 0:0 + 16]),
                layout=((16, 16), (33, 1)) + 528,
                shape=(16, 16),
            )
            T.copy(value_183, window_13)
            mbb = T.tensor_view(
                62848, dtype="f32", storage="smem", layout=(32, 33), shape=(32, 33)
            )
            T.fill(mbb, 0.0)
            value_185 = T.tensor_view(
                T.ptr_of(asm_12[32:32 + 1, 32:32 + 1]),
                layout=((16, 16), (65, 1)),
                shape=(16, 16),
            )
            value_186 = T.tensor_view(T.ptr_of(mbb), layout=(32, 33), shape=(32, 33))
            window_14 = T.tensor_view(
                T.ptr_of(value_186[0:0 + 16, 0:0 + 16]),
                layout=((16, 16), (33, 1)),
                shape=(16, 16),
            )
            T.copy(value_185, window_14)
            value_187 = T.tensor_view(
                T.ptr_of(asm_12[48:48 + 1, 48:48 + 1]),
                layout=((16, 16), (65, 1)),
                shape=(16, 16),
            )
            value_188 = T.tensor_view(T.ptr_of(mbb), layout=(32, 33), shape=(32, 33))
            window_15 = T.tensor_view(
                T.ptr_of(value_188[16:16 + 16, 16:16 + 16]),
                layout=((16, 16), (33, 1)) + 544,
                shape=(16, 16),
            )
            T.copy(value_187, window_15)
            value_189 = T.tensor_view(
                T.ptr_of(asm_12[32:32 + 1, 48:48 + 1]),
                layout=((16, 16), (1, 65)),
                shape=(16, 16),
            )
            value_190 = T.tensor_view(T.ptr_of(mbb), layout=(32, 33), shape=(32, 33))
            window_16 = T.tensor_view(
                T.ptr_of(value_190[16:16 + 16, 0:0 + 16]),
                layout=((16, 16), (33, 1)) + 528,
                shape=(16, 16),
            )
            T.copy(value_189, window_16)
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_11:
                value_191 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1), "rmem"
                    ]
                )
                value_192 = T.tensor_view(
                    67072, dtype="f32", storage="smem", layout=(32, 33), shape=(32, 33)
                )
                T.fill(value_192, 0.0)
                value_193 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1), "rmem"
                    ]
                )
                asm_13 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_194 = T.tensor_view(
                    T.ptr_of(asm_13[32:32 + 1, 0:0 + 1]),
                    layout=((32, 32, 1), (65, 1, 0)),
                    shape=(32, 32, 1),
                )
                value_195 = T.tensor_view(
                    T.ptr_of(value_193),
                    layout=(4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1),
                    shape=(32, 32, 1),
                )
                T.copy(value_194, value_195)
                value_196 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1), "rmem"
                    ]
                )
                value_197 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 32),
                        "f32",
                        ((4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 32), (64, 0, 32, 0, 1)),
                        "rmem",
                    ]
                )
                value_198 = T.alloc_tensor(
                    tensor_type=Tensor[(32, 1, 32), "f32", (4, 8 @ scope_11.d0, 1, 32), "rmem"]
                )
                value_199 = T.tensor_view(
                    T.ptr_of(asm_13[32:32 + 1, 0:0 + 1]),
                    layout=((32, 1, 32), (65, 0, 1)),
                    shape=(32, 1, 32),
                )
                value_200 = T.tensor_view(
                    T.ptr_of(value_198), layout=(4, 8 @ scope_11.d0, 1, 32), shape=(32, 1, 32)
                )
                T.copy(value_199, value_200)
                value_201 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32, 32), "f32", (1, 2, 16 @ scope_11.d1, 32), "rmem"]
                )
                maa_2 = T.tensor_view(
                    58624, dtype="f32", storage="smem", layout=(32, 33), shape=(32, 33)
                )
                value_202 = T.tensor_view(
                    T.ptr_of(maa_2), layout=((1, 32, 32), (0, 1, 33)), shape=(1, 32, 32)
                )
                value_203 = T.tensor_view(
                    T.ptr_of(value_201), layout=(1, 2, 16 @ scope_11.d1, 32), shape=(1, 32, 32)
                )
                T.copy(value_202, value_203)
                value_204 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 32),
                        "f32",
                        ((8 @ scope_11.d0, 4, 16 @ scope_11.d1, 2, 32), (4096, 1024, 64, 32, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_200, value_203, value_204, kind=BinaryKind.MUL)
                value_205 = T.tensor_view(
                    T.ptr_of(value_197),
                    layout=((4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 32), (64, 0, 32, 0, 1)),
                    shape=(32, 32, 32),
                )
                T.copy(value_204, value_205)
                value_206 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1),
                        "f32",
                        ((4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1), (256, 32, 16, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_205, value_206, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_207 = T.tensor_view(
                    T.ptr_of(value_196),
                    layout=(4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1),
                    shape=(32, 32, 1),
                )
                T.copy(value_206, value_207)
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1, 1), "rmem"
                    ]
                )
                T.binary(value_195, value_207, y, kind=BinaryKind.ADD)
                value_208 = T.tensor_view(
                    T.ptr_of(y),
                    layout=(4, 8 @ scope_11.d0, 2, 16 @ scope_11.d1),
                    shape=(32, 32),
                )
                value_209 = T.tensor_view(T.ptr_of(value_192), layout=(32, 33), shape=(32, 33))
                window_17 = T.tensor_view(
                    T.ptr_of(value_209[0:0 + 32, 0:0 + 32]),
                    layout=((32, 32), (33, 1)),
                    shape=(32, 32),
                )
                T.copy(value_208, window_17)
                value_210 = T.tensor_view(
                    T.ptr_of(value_192), layout=((32, 32, 1), (1, 33, 0)), shape=(32, 32, 1)
                )
                value_211 = T.tensor_view(
                    T.ptr_of(value_191),
                    layout=(2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1),
                    shape=(32, 32, 1),
                )
                T.copy(value_210, value_211)
                value_212 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1), "rmem"
                    ]
                )
                value_213 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 32),
                        "f32",
                        ((2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 32), (128, 0, 32, 0, 1)),
                        "rmem",
                    ]
                )
                value_214 = T.alloc_tensor(
                    tensor_type=Tensor[(32, 1, 32), "f32", (2, 16 @ scope_11.d1, 1, 32), "rmem"]
                )
                value_215 = T.tensor_view(
                    T.ptr_of(value_192), layout=((32, 1, 32), (1, 0, 33)), shape=(32, 1, 32)
                )
                value_216 = T.tensor_view(
                    T.ptr_of(value_214), layout=(2, 16 @ scope_11.d1, 1, 32), shape=(32, 1, 32)
                )
                T.copy(value_215, value_216)
                value_217 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32, 32), "f32", (1, 4, 8 @ scope_11.d0, 32), "rmem"]
                )
                mbb_2 = T.tensor_view(
                    62848, dtype="f32", storage="smem", layout=(32, 33), shape=(32, 33)
                )
                value_218 = T.tensor_view(
                    T.ptr_of(mbb_2), layout=((1, 32, 32), (0, 33, 1)), shape=(1, 32, 32)
                )
                value_219 = T.tensor_view(
                    T.ptr_of(value_217), layout=(1, 4, 8 @ scope_11.d0, 32), shape=(1, 32, 32)
                )
                T.copy(value_218, value_219)
                value_220 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 32),
                        "f32",
                        ((16 @ scope_11.d1, 2, 8 @ scope_11.d0, 4, 32), (2048, 1024, 128, 32, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_216, value_219, value_220, kind=BinaryKind.MUL)
                value_221 = T.tensor_view(
                    T.ptr_of(value_213),
                    layout=((2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 32), (128, 0, 32, 0, 1)),
                    shape=(32, 32, 32),
                )
                T.copy(value_220, value_221)
                value_222 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1),
                        "f32",
                        ((2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1), (512, 32, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.reduce(value_221, value_222, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
                value_223 = T.tensor_view(
                    T.ptr_of(value_212),
                    layout=(2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1),
                    shape=(32, 32, 1),
                )
                T.copy(value_222, value_223)
                tbat = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 1), "f32", (2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0, 1), "rmem"
                    ]
                )
                T.binary(value_211, value_223, tbat, kind=BinaryKind.ADD)
                value_224 = T.tensor_view(
                    T.ptr_of(tbat),
                    layout=(2, 16 @ scope_11.d1, 4, 8 @ scope_11.d0),
                    shape=(32, 32),
                )
                value_225 = T.tensor_view(T.ptr_of(asm_13), layout=(64, 65), shape=(64, 65))
                window_18 = T.tensor_view(
                    T.ptr_of(value_225[0:0 + 32, 32:32 + 32]),
                    layout=((32, 32), (65, 1)) + 32,
                    shape=(32, 32),
                )
                T.copy(value_224, window_18)
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_12:
                value_227 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_228 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "i32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                value_229 = T.tensor_view(
                    T.ptr_of(iota), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_230 = T.tensor_view(
                    T.ptr_of(value_228),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 1), (16, 8, 1, 0)),
                    shape=(64, 1),
                )
                T.copy(value_229, value_230)
                value_231 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "i32", ((1, 2, 4 @ scope_12.d2, 8), (0, 64, 128, 512)), "rmem"
                    ]
                )
                value_232 = T.tensor_view(
                    T.ptr_of(iota), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_233 = T.tensor_view(
                    T.ptr_of(value_231),
                    layout=((1, 2, 4 @ scope_12.d2, 8), (0, 64, 128, 512)),
                    shape=(1, 64),
                )
                T.copy(value_232, value_233)
                value_234 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_12.d0, 8 @ scope_12.d1, 2, 4 @ scope_12.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_230, value_233, value_234, kind=BinaryKind.SUB)
                value_235 = T.tensor_view(
                    T.ptr_of(value_227),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_234, value_235)
                zi2 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
                T.fill(zi2, 0.0)
                value_236 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bool",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_235, zi2, value_236, kind=BinaryKind.EQ)
                value_237 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bool",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_235, zi2, value_237, kind=BinaryKind.GT)
                value_238 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_239 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                T.fill(value_239, 0.0)
                value_240 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 16.0)
                T.binary(value_239, scalar_3, value_240, kind=BinaryKind.ADD)
                c16 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
                T.cast(value_240, c16, dtype="i32")
                value_241 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "i32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 1), (16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.binary(value_230, c16, value_241, kind=BinaryKind.FLOOR_DIV)
                value_242 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "i32", ((1, 2, 4 @ scope_12.d2, 8), (0, 64, 128, 512)), "rmem"
                    ]
                )
                T.binary(value_233, c16, value_242, kind=BinaryKind.FLOOR_DIV)
                value_243 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "i32",
                        ((4 @ scope_12.d0, 8 @ scope_12.d1, 2, 4 @ scope_12.d2, 16), (1024, 128, 64, 16, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_241, value_242, value_243, kind=BinaryKind.SUB)
                value_244 = T.tensor_view(
                    T.ptr_of(value_238),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_243, value_244)
                value_245 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bool",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_244, zi2, value_245, kind=BinaryKind.EQ)
                value_246 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                asm_14 = T.tensor_view(
                    41984, dtype="f32", storage="smem", layout=(64, 65), shape=(64, 65)
                )
                value_247 = T.tensor_view(
                    T.ptr_of(asm_14), layout=((64, 64), (65, 1)), shape=(64, 64)
                )
                value_248 = T.tensor_view(
                    T.ptr_of(value_246),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_247, value_248)
                value_249 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_250 = T.tensor_view(
                    T.ptr_of(asm_14), layout=((64, 64), (1, 65)), shape=(64, 64)
                )
                value_251 = T.tensor_view(
                    T.ptr_of(value_249),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_250, value_251)
                value_252 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.where(value_245, value_248, value_251, value_252)
                value_253 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_4, 0.0)
                T.where(value_237, value_252, scalar_4, value_253)
                tm = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_5, 1.0)
                T.where(value_236, scalar_5, value_253, tm)
                value_254 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 8, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(tm, value_254, dtype="bf16")
                value_226 = T.tensor_view(
                    41984,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_255 = T.tensor_view(
                    T.ptr_of(value_226),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_254, value_255)
            with scope[:1] as scope_13:
                value_257 = T.tensor_view(
                    T.ptr_of(v[cta.d0 * 64:cta.d0 * 64 + 1, cta.d1:cta.d1 + 1, 0:0 + 1]),
                    layout=((64, 128), (6144, 1)),
                    shape=(64, 128),
                )
                value_256 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_258 = T.tensor_view(
                    T.ptr_of(value_256),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_257, value_258)
            value_259 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            sx = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xv = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_14:
                gb_3 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                value_260 = T.tensor_view(
                    T.ptr_of(gb_3[1:1 + 1, 0:0 + 1]),
                    layout=((8, 8, 1, 1), (8, 1, 0, 0)),
                    shape=(8, 8, 1, 1),
                )
                value_261 = T.tensor_view(
                    T.ptr_of(value_259), layout=(8, 8 @ scope_14.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_260, value_261)
                value_262 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_14.d0, 16 @ scope_14.d1, 4), "rmem"
                    ]
                )
                value_256_2 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_263 = T.tensor_view(
                    T.ptr_of(value_256_2),
                    layout=(8, 8, 16, 4) | Swizzle(3, 4, 3),
                    shape=(8, 8, 16, 4),
                )
                value_264 = T.tensor_view(
                    T.ptr_of(value_262),
                    layout=(8, 8 @ scope_14.d0, 16 @ scope_14.d1, 4),
                    shape=(8, 8, 16, 4),
                )
                T.copy(value_263, value_264)
                T.cast(value_264, sx, dtype="f32")
                T.binary(sx, value_261, xv, kind=BinaryKind.MUL)
                value_266 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_14.d0, 16 @ scope_14.d1, 4), "rmem"
                    ]
                )
                T.cast(xv, value_266, dtype="bf16")
                value_267 = T.tensor_view(
                    T.ptr_of(value_266),
                    layout=(8, 8 @ scope_14.d0, 16 @ scope_14.d1, 4),
                    shape=(64, 64),
                )
                value_265 = T.tensor_view(
                    66560,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_268 = T.tensor_view(
                    T.ptr_of(value_265),
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_267, value_268)
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_15:
                value_269 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_15.d0, 2, 8 @ scope_15.d1, 8, 4 @ scope_15.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_269, 0.0)
                for o_m_2 in range(0, 64, 64):
                    for o_n_2 in range(0, 64, 64):
                        for o_k_2 in range(0, 64, 64):
                            value_226_2 = T.tensor_view(
                                41984,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_270 = T.tensor_view(
                                T.ptr_of(value_226_2),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_8 = T.tensor_view(
                                T.ptr_of(value_270[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_265_2 = T.tensor_view(
                                66560,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_271 = T.tensor_view(
                                T.ptr_of(value_265_2),
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_8 = T.tensor_view(
                                T.ptr_of(value_271[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_269,
                                lhs_view_8,
                                rhs_view_8,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_9 = T.tensor_view(
                                T.ptr_of(value_270[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_9 = T.tensor_view(
                                T.ptr_of(value_271[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_269,
                                lhs_view_9,
                                rhs_view_9,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_10 = T.tensor_view(
                                T.ptr_of(value_270[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_10 = T.tensor_view(
                                T.ptr_of(value_271[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_269,
                                lhs_view_10,
                                rhs_view_10,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_11 = T.tensor_view(
                                T.ptr_of(value_270[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_11 = T.tensor_view(
                                T.ptr_of(value_271[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_269,
                                lhs_view_11,
                                rhs_view_11,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_272 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_15.d0, 2, 8 @ scope_15.d1, 8, 4 @ scope_15.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_273 = T.tensor_view(
                    T.ptr_of(value_269),
                    layout=((4 @ scope_15.d0, 2, 8 @ scope_15.d1, 8, 4 @ scope_15.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(value_273, value_272, dtype="bf16")
                ost_2 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_274 = T.tensor_view(
                    T.ptr_of(ost_2),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                window_19 = T.tensor_view(
                    T.ptr_of(value_274[0:0 + 64, 0:0 + 64]),
                    layout=Layout((64, (64,)), (64, (1,))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_272, window_19)
            value_275 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            sx_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xv_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_16:
                gb_4 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                value_276 = T.tensor_view(
                    T.ptr_of(gb_4[1:1 + 1, 0:0 + 1]),
                    layout=((8, 8, 1, 1), (8, 1, 0, 0)),
                    shape=(8, 8, 1, 1),
                )
                value_277 = T.tensor_view(
                    T.ptr_of(value_275), layout=(8, 8 @ scope_16.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_276, value_277)
                value_278 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_16.d0, 16 @ scope_16.d1, 4), "rmem"
                    ]
                )
                value_256_3 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_279 = T.tensor_view(
                    T.ptr_of(value_256_3),
                    layout=(8, 8, 16, 4) + 4096 | Swizzle(3, 4, 3),
                    shape=(8, 8, 16, 4),
                )
                value_280 = T.tensor_view(
                    T.ptr_of(value_278),
                    layout=(8, 8 @ scope_16.d0, 16 @ scope_16.d1, 4),
                    shape=(8, 8, 16, 4),
                )
                T.copy(value_279, value_280)
                T.cast(value_280, sx_1, dtype="f32")
                T.binary(sx_1, value_277, xv_1, kind=BinaryKind.MUL)
                value_282 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_16.d0, 16 @ scope_16.d1, 4), "rmem"
                    ]
                )
                T.cast(xv_1, value_282, dtype="bf16")
                value_283 = T.tensor_view(
                    T.ptr_of(value_282),
                    layout=(8, 8 @ scope_16.d0, 16 @ scope_16.d1, 4),
                    shape=(64, 64),
                )
                value_281 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_284 = T.tensor_view(
                    T.ptr_of(value_281),
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_283, value_284)
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_17:
                value_285 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_17.d0, 2, 8 @ scope_17.d1, 8, 4 @ scope_17.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_285, 0.0)
                for o_m_3 in range(0, 64, 64):
                    for o_n_3 in range(0, 64, 64):
                        for o_k_3 in range(0, 64, 64):
                            value_226_3 = T.tensor_view(
                                41984,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_286 = T.tensor_view(
                                T.ptr_of(value_226_3),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_12 = T.tensor_view(
                                T.ptr_of(value_286[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_281_2 = T.tensor_view(
                                50176,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_287 = T.tensor_view(
                                T.ptr_of(value_281_2),
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_12 = T.tensor_view(
                                T.ptr_of(value_287[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_285,
                                lhs_view_12,
                                rhs_view_12,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_13 = T.tensor_view(
                                T.ptr_of(value_286[o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_13 = T.tensor_view(
                                T.ptr_of(value_287[o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_285,
                                lhs_view_13,
                                rhs_view_13,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_14 = T.tensor_view(
                                T.ptr_of(value_286[o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_14 = T.tensor_view(
                                T.ptr_of(value_287[o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_285,
                                lhs_view_14,
                                rhs_view_14,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_15 = T.tensor_view(
                                T.ptr_of(value_286[o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_15 = T.tensor_view(
                                T.ptr_of(value_287[o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_285,
                                lhs_view_15,
                                rhs_view_15,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_288 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_17.d0, 2, 8 @ scope_17.d1, 8, 4 @ scope_17.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_289 = T.tensor_view(
                    T.ptr_of(value_285),
                    layout=((4 @ scope_17.d0, 2, 8 @ scope_17.d1, 8, 4 @ scope_17.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(value_289, value_288, dtype="bf16")
                ost_3 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_290 = T.tensor_view(
                    T.ptr_of(ost_3),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                window_20 = T.tensor_view(
                    T.ptr_of(value_290[0:0 + 64, 64:64 + 64]),
                    layout=Layout((64, (64,)), (64, (1,))) + 4096 | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_288, window_20)
            ost_1 = T.tensor_view(
                50176,
                dtype="bf16",
                storage="smem",
                layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                shape=(64, 128),
            )
            T.fill(ost_1, 0.0)
            value_291 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            sx_2 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_297 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xv_2 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_18:
                gb_5 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                value_292 = T.tensor_view(
                    T.ptr_of(gb_5[1:1 + 1, 0:0 + 1]),
                    layout=((8, 8, 1, 1), (8, 1, 0, 0)),
                    shape=(8, 8, 1, 1),
                )
                value_293 = T.tensor_view(
                    T.ptr_of(value_291), layout=(8, 8 @ scope_18.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_292, value_293)
                value_294 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_18.d0, 16 @ scope_18.d1, 4), "rmem"
                    ]
                )
                value_8_4 = T.tensor_view(
                    17408,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_295 = T.tensor_view(
                    T.ptr_of(value_8_4),
                    layout=(8, 8, 16, 4) | Swizzle(3, 4, 3),
                    shape=(8, 8, 16, 4),
                )
                value_296 = T.tensor_view(
                    T.ptr_of(value_294),
                    layout=(8, 8 @ scope_18.d0, 16 @ scope_18.d1, 4),
                    shape=(8, 8, 16, 4),
                )
                T.copy(value_295, value_296)
                T.cast(value_296, sx_2, dtype="f32")
                value_298 = T.tensor_view(
                    T.ptr_of(gb_5), layout=((8, 8, 1, 1), (8, 1, 0, 0)), shape=(8, 8, 1, 1)
                )
                value_299 = T.tensor_view(
                    T.ptr_of(value_297), layout=(8, 8 @ scope_18.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_298, value_299)
                value_300 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "f32", (8, 8 @ scope_18.d0, 16 @ scope_18.d1, 4), "rmem"
                    ]
                )
                T.binary(sx_2, value_293, value_300, kind=BinaryKind.MUL)
                value_301 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8, 8 @ scope_18.d0, 1, 1), "rmem"]
                )
                T.unary(value_299, value_301, kind=UnaryKind.EXP)
                T.binary(value_300, value_301, xv_2, kind=BinaryKind.MUL)
                value_303 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_18.d0, 16 @ scope_18.d1, 4), "rmem"
                    ]
                )
                T.cast(xv_2, value_303, dtype="bf16")
                value_304 = T.tensor_view(
                    T.ptr_of(value_303),
                    layout=(8, 8 @ scope_18.d0, 16 @ scope_18.d1, 4),
                    shape=(64, 64),
                )
                value_302 = T.tensor_view(
                    66560,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_305 = T.tensor_view(
                    T.ptr_of(value_302),
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_304, value_305)
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_19:
                value_306 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_19.d0, 2, 8 @ scope_19.d1, 8, 4 @ scope_19.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_306, 0.0)
                for o_m_4 in range(0, 64, 64):
                    for o_n_4 in range(0, 64, 64):
                        for o_k_4 in range(0, 64, 64):
                            value_226_4 = T.tensor_view(
                                41984,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_307 = T.tensor_view(
                                T.ptr_of(value_226_4),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_16 = T.tensor_view(
                                T.ptr_of(value_307[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_302_2 = T.tensor_view(
                                66560,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_308 = T.tensor_view(
                                T.ptr_of(value_302_2),
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_16 = T.tensor_view(
                                T.ptr_of(value_308[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_306,
                                lhs_view_16,
                                rhs_view_16,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_17 = T.tensor_view(
                                T.ptr_of(value_307[o_m_4:o_m_4 + 64, o_k_4 + 16:o_k_4 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_17 = T.tensor_view(
                                T.ptr_of(value_308[o_k_4 + 16:o_k_4 + 16 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_306,
                                lhs_view_17,
                                rhs_view_17,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_18 = T.tensor_view(
                                T.ptr_of(value_307[o_m_4:o_m_4 + 64, o_k_4 + 32:o_k_4 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_18 = T.tensor_view(
                                T.ptr_of(value_308[o_k_4 + 32:o_k_4 + 32 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_306,
                                lhs_view_18,
                                rhs_view_18,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_19 = T.tensor_view(
                                T.ptr_of(value_307[o_m_4:o_m_4 + 64, o_k_4 + 48:o_k_4 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_19 = T.tensor_view(
                                T.ptr_of(value_308[o_k_4 + 48:o_k_4 + 48 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_306,
                                lhs_view_19,
                                rhs_view_19,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_309 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_19.d0, 2, 8 @ scope_19.d1, 8, 4 @ scope_19.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_310 = T.tensor_view(
                    T.ptr_of(value_306),
                    layout=((4 @ scope_19.d0, 2, 8 @ scope_19.d1, 8, 4 @ scope_19.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(value_310, value_309, dtype="bf16")
                ost_1_2 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_311 = T.tensor_view(
                    T.ptr_of(ost_1_2),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                window_21 = T.tensor_view(
                    T.ptr_of(value_311[0:0 + 64, 0:0 + 64]),
                    layout=Layout((64, (64,)), (64, (1,))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_309, window_21)
            value_312 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            sx_3 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_318 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 1, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 1, 1), None),
                        attrs=(S(1), B()),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xv_3 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 8, 16, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 8, 16, 4), None),
                        attrs=(S(1), S(2)),
                        mesh=Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")
            ) as scope_20:
                gb_6 = T.tensor_view(
                    16384, dtype="f32", storage="smem", layout=(4, 64), shape=(4, 64)
                )
                value_313 = T.tensor_view(
                    T.ptr_of(gb_6[1:1 + 1, 0:0 + 1]),
                    layout=((8, 8, 1, 1), (8, 1, 0, 0)),
                    shape=(8, 8, 1, 1),
                )
                value_314 = T.tensor_view(
                    T.ptr_of(value_312), layout=(8, 8 @ scope_20.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_313, value_314)
                value_315 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_20.d0, 16 @ scope_20.d1, 4), "rmem"
                    ]
                )
                value_8_5 = T.tensor_view(
                    17408,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_316 = T.tensor_view(
                    T.ptr_of(value_8_5),
                    layout=(8, 8, 16, 4) + 4096 | Swizzle(3, 4, 3),
                    shape=(8, 8, 16, 4),
                )
                value_317 = T.tensor_view(
                    T.ptr_of(value_315),
                    layout=(8, 8 @ scope_20.d0, 16 @ scope_20.d1, 4),
                    shape=(8, 8, 16, 4),
                )
                T.copy(value_316, value_317)
                T.cast(value_317, sx_3, dtype="f32")
                value_319 = T.tensor_view(
                    T.ptr_of(gb_6), layout=((8, 8, 1, 1), (8, 1, 0, 0)), shape=(8, 8, 1, 1)
                )
                value_320 = T.tensor_view(
                    T.ptr_of(value_318), layout=(8, 8 @ scope_20.d0, 1, 1), shape=(8, 8, 1, 1)
                )
                T.copy(value_319, value_320)
                value_321 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "f32", (8, 8 @ scope_20.d0, 16 @ scope_20.d1, 4), "rmem"
                    ]
                )
                T.binary(sx_3, value_314, value_321, kind=BinaryKind.MUL)
                value_322 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 8, 1, 1), "f32", (8, 8 @ scope_20.d0, 1, 1), "rmem"]
                )
                T.unary(value_320, value_322, kind=UnaryKind.EXP)
                T.binary(value_321, value_322, xv_3, kind=BinaryKind.MUL)
                value_324 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 4), "bf16", (8, 8 @ scope_20.d0, 16 @ scope_20.d1, 4), "rmem"
                    ]
                )
                T.cast(xv_3, value_324, dtype="bf16")
                value_325 = T.tensor_view(
                    T.ptr_of(value_324),
                    layout=(8, 8 @ scope_20.d0, 16 @ scope_20.d1, 4),
                    shape=(64, 64),
                )
                value_323 = T.tensor_view(
                    16384,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_326 = T.tensor_view(
                    T.ptr_of(value_323),
                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_325, value_326)
            with Mesh(
                (Topology("thread", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_21:
                value_327 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_21.d0, 2, 8 @ scope_21.d1, 8, 4 @ scope_21.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_327, 0.0)
                for o_m_5 in range(0, 64, 64):
                    for o_n_5 in range(0, 64, 64):
                        for o_k_5 in range(0, 64, 64):
                            value_226_5 = T.tensor_view(
                                41984,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_328 = T.tensor_view(
                                T.ptr_of(value_226_5),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_20 = T.tensor_view(
                                T.ptr_of(value_328[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_323_2 = T.tensor_view(
                                16384,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_329 = T.tensor_view(
                                T.ptr_of(value_323_2),
                                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_20 = T.tensor_view(
                                T.ptr_of(value_329[o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_327,
                                lhs_view_20,
                                rhs_view_20,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_21 = T.tensor_view(
                                T.ptr_of(value_328[o_m_5:o_m_5 + 64, o_k_5 + 16:o_k_5 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_21 = T.tensor_view(
                                T.ptr_of(value_329[o_k_5 + 16:o_k_5 + 16 + 16, o_n_5:o_n_5 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_327,
                                lhs_view_21,
                                rhs_view_21,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_22 = T.tensor_view(
                                T.ptr_of(value_328[o_m_5:o_m_5 + 64, o_k_5 + 32:o_k_5 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_22 = T.tensor_view(
                                T.ptr_of(value_329[o_k_5 + 32:o_k_5 + 32 + 16, o_n_5:o_n_5 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_327,
                                lhs_view_22,
                                rhs_view_22,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_23 = T.tensor_view(
                                T.ptr_of(value_328[o_m_5:o_m_5 + 64, o_k_5 + 48:o_k_5 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_23 = T.tensor_view(
                                T.ptr_of(value_329[o_k_5 + 48:o_k_5 + 48 + 16, o_n_5:o_n_5 + 64]),
                                layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_327,
                                lhs_view_23,
                                rhs_view_23,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_330 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_21.d0, 2, 8 @ scope_21.d1, 8, 4 @ scope_21.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_331 = T.tensor_view(
                    T.ptr_of(value_327),
                    layout=((4 @ scope_21.d0, 2, 8 @ scope_21.d1, 8, 4 @ scope_21.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.cast(value_331, value_330, dtype="bf16")
                ost_1_3 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_332 = T.tensor_view(
                    T.ptr_of(ost_1_3),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                window_22 = T.tensor_view(
                    T.ptr_of(value_332[0:0 + 64, 64:64 + 64]),
                    layout=Layout((64, (64,)), (64, (1,))) + 4096 | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_330, window_22)
            T.fill(out, 0.0)
            with scope[:1] as scope_22:
                value_20_2 = T.tensor_view(
                    33792,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_334 = T.tensor_view(
                    T.ptr_of(value_20_2),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_333_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
                )
                value_335 = T.tensor_view(
                    T.ptr_of(value_333_1), layout=(64, 64), shape=(64, 64)
                )
                T.copy_async_tensor(value_334, value_335)
                value_336 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 15360), shape=(1024, 15360)
                )
                window_23 = T.tensor_view(
                    T.ptr_of(value_336[cta.d0 * 64:cta.d0 * 64 + 64, (cta.d1 * 320) + (2 * 128):(cta.d1 * 320) + (2 * 128) + 64]),
                    layout=((64, 64), (15360, 1)),
                    shape=(64, 64),
                )
                T.copy(value_335, window_23)
            with scope[:1] as scope_23:
                ost_1_4 = T.tensor_view(
                    50176,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_338 = T.tensor_view(
                    T.ptr_of(ost_1_4),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_337_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
                )
                value_339 = T.tensor_view(
                    T.ptr_of(value_337_1), layout=(64, 128), shape=(64, 128)
                )
                T.copy_async_tensor(value_338, value_339)
                value_340 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 15360), shape=(1024, 15360)
                )
                window_24 = T.tensor_view(
                    T.ptr_of(value_340[cta.d0 * 64:cta.d0 * 64 + 64, cta.d1 * 320:cta.d1 * 320 + 128]),
                    layout=((64, 128), (15360, 1)),
                    shape=(64, 128),
                )
                T.copy(value_339, window_24)
            with scope[:1] as scope_24:
                ost_4 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_342 = T.tensor_view(
                    T.ptr_of(ost_4),
                    layout=Layout((64, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_341_1 = T.tensor_view(
                    T.ptr_of(scratch_2[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
                )
                value_343 = T.tensor_view(
                    T.ptr_of(value_341_1), layout=(64, 128), shape=(64, 128)
                )
                T.copy_async_tensor(value_342, value_343)
                value_344 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 15360), shape=(1024, 15360)
                )
                window_25 = T.tensor_view(
                    T.ptr_of(value_344[cta.d0 * 64:cta.d0 * 64 + 64, (cta.d1 * 320) + 128:(cta.d1 * 320) + 128 + 128]),
                    layout=((64, 128), (15360, 1)),
                    shape=(64, 128),
                )
                T.copy(value_343, window_25)
