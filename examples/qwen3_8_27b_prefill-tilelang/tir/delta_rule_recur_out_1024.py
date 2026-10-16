# analysis target=nvidia.h200_sxm module=DeltaRuleRecur function=recur_out topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r1.13MB/w12.25MB@logical,r108.38MB/w1.15GB@total,r1.13MB/w12.25MB@cta,r1.13MB/w12.25MB@thread;rmem:r4.78MB/w3.59MB@logical,r459.33MB/w389.64MB@total,r4.78MB/w4.06MB@cta,r42.16KB/w33.02KB@thread;smem:r1.89MB/w1.64MB@logical,r181.14MB/w157.12MB@total,r1.89MB/w1.64MB@cta,r349.44KB/w1.01MB@thread footprint=gc:12.00KB;k:256.00KB;q:256.00KB;v95:348:8.00KB;v97:349:768.00KB;wup:1.88MB footprint-precision=exact peak=gmem:50.20MB;rmem:64.27KB;smem:226.00KB persistent=gmem:38.19MB
#   buffer=wup holds=3.14MB time=none space=cta.vb reuse=1.88MB fits=yes precision=exact
#   buffer=k holds=3.14MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=q holds=3.14MB time=none space=cta.vb reuse=256.00KB fits=yes precision=exact
#   buffer=gc holds=3.14MB time=none space=cta.vb reuse=12.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def recur_out(
    q: Tensor[(1024, 16, 128), "bf16"], k: Tensor[(1024, 16, 128), "bf16"], wup: Tensor[(1024, 48, 320), "bf16"], gc: Tensor[(48, 1024), "f32"], scratch: Tensor[(4096,), "bf16"], out: Tensor[(1024, 48, 128), "bf16"]
):
    with Mesh((Topology("cta", 96),), layout=(2, 48), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            st = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 8, 4, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        attrs=(S(1), S(3), S(5)),
                        mesh=Mesh((Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
            ) as scope_1:
                T.fill(st, 0.0)
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(49152, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_3_stages = (
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_6_stages = (
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            value_9_stages = (
                T.tensor_view(172032, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_12_stages = (
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(196608, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_15_stages = (
                T.tensor_view(222720, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
                T.tensor_view(222976, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
            )
            value_18_stages = (
                T.tensor_view(221184, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221440, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221696, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            value_28_stages = (
                T.tensor_view(221952, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222208, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222464, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            value_32_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            value_56_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(155648, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            value_59_stages = (
                T.tensor_view(204800, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(212992, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            for c in range(0, 16, 1):
                with scope[2:, :32] as scope_2:
                    value_1 = T.tensor_view(
                        T.ptr_of(wup[c * 64:c * 64 + 1, cta.d1:cta.d1 + 1, 0:0 + 1]),
                        layout=((64, 128), (15360, 1)),
                        shape=(64, 128),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(value_stages[c % 2]),
                        layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_1, value_2)
                    value_4 = T.tensor_view(
                        T.ptr_of(k[c * 64:c * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(value_3_stages[c % 2]),
                        layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 128),
                    )
                    T.copy_async_tensor(value_4, value_5)
                    value_7 = T.tensor_view(
                        T.ptr_of(q[c * 64:c * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
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
                        T.ptr_of(wup[c * 64:c * 64 + 1, cta.d1:cta.d1 + 1, 128 + (cta.d0 * 64):128 + (cta.d0 * 64) + 1]),
                        layout=((64, 64), (15360, 1)),
                        shape=(64, 64),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_9_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_10, value_11)
                    value_13 = T.tensor_view(
                        T.ptr_of(wup[c * 64:c * 64 + 1, cta.d1:cta.d1 + 1, 256:256 + 1]),
                        layout=((64, 64), (15360, 1)),
                        shape=(64, 64),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_12_stages[c % 2]),
                        layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy_async_tensor(value_13, value_14)
                    value_16 = T.tensor_view(
                        T.ptr_of(gc[cta.d1:cta.d1 + 1, c * 64:c * 64 + 1]),
                        layout=((1, 64), (0, 1)),
                        shape=(1, 64),
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2]), layout=(1, 64), shape=(1, 64)
                    )
                    T.copy_async_tensor(value_16, value_17)
                with Mesh((Topology("thread", 384),), layout=(64,), names=("d0",)) as scope_3:
                    value_19 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                    value_20 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2][0:0 + 1, 63:63 + 1]),
                        layout=((1,), (0,)),
                        shape=(1,),
                    )
                    value_21 = T.tensor_view(T.ptr_of(value_19), layout=(1,), shape=(1,))
                    T.copy(value_20, value_21)
                    value_22 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.d0,), "rmem"]
                    )
                    value_23 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2]), layout=(64,), shape=(64,)
                    )
                    value_24 = T.tensor_view(
                        T.ptr_of(value_22), layout=(64 @ scope_3.d0,), shape=(64,)
                    )
                    T.copy(value_23, value_24)
                    value_25 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.d0,), "rmem"]
                    )
                    T.binary(value_21, value_24, value_25, kind=BinaryKind.SUB)
                    value_26 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.d0,), "rmem"]
                    )
                    T.unary(value_25, value_26, kind=UnaryKind.EXP)
                    value_27 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    T.copy(value_26, value_27)
                    value_29 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.d0,), "rmem"]
                    )
                    T.unary(value_24, value_29, kind=UnaryKind.EXP)
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[(64,), "f32", (64 @ scope_3.d0,), "rmem"]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 0.08838834764831845)
                    T.binary(value_29, scalar, value_30, kind=BinaryKind.MUL)
                    value_31 = T.tensor_view(
                        T.ptr_of(value_28_stages[c % 3]), layout=(64,), shape=(64,)
                    )
                    T.copy(value_30, value_31)
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
                ) as scope_4:
                    value_33 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 64),
                            "bf16",
                            ((2, 4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (4096, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_34 = T.tensor_view(
                        T.ptr_of(st),
                        layout=((2, 4 @ scope_4.d0, 2, 8 @ scope_4.d1, 8, 4 @ scope_4.d2, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    T.cast(value_34, value_33, dtype="bf16")
                    value_35 = T.tensor_view(
                        T.ptr_of(value_32_stages[c % 3]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy(value_33, value_35)
                with Mesh(
                    (Topology("thread", 384),), layout=(4, 8, 4), names=("d0", "d1", "d2")
                ) as scope_5:
                    value_36 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    value_37 = T.tensor_view(
                        T.ptr_of(value_15_stages[c % 2][0:0 + 1, 63:63 + 1]),
                        layout=((1, 1), (0, 0)),
                        shape=(1, 1),
                    )
                    value_38 = T.tensor_view(T.ptr_of(value_36), layout=(1, 1), shape=(1, 1))
                    T.copy(value_37, value_38)
                    el = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                    T.unary(value_38, el, kind=UnaryKind.EXP)
                    value_39 = T.tensor_view(
                        T.ptr_of(st),
                        layout=((2, 4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (4096, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    T.binary(value_39, el, value_39, kind=BinaryKind.MUL)
                    value_41 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_42 = T.tensor_view(
                        T.ptr_of(value_9_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_43 = T.tensor_view(
                        T.ptr_of(value_41),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_42, value_43)
                    uu = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_43, uu, dtype="f32")
                    vn = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(vn, 0.0)
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 64, 64):
                            for o_k in range(0, 128, 64):
                                value_44 = T.tensor_view(
                                    T.ptr_of(value_stages[c % 2]),
                                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 128),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_44[o_m:o_m + 64, o_k:o_k + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_45 = T.tensor_view(
                                    T.ptr_of(value_32_stages[c % 3]),
                                    layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_45[o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    vn,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_44[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_45[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    vn,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_44[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_45[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    vn,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_44[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_45[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    vn,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    vn_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_46 = T.tensor_view(
                        T.ptr_of(vn),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.binary(uu, value_46, vn_1, kind=BinaryKind.SUB)
                    value_47 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1), (16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(value_18_stages[c % 3]),
                        layout=((64, 1), (1, 0)),
                        shape=(64, 1),
                    )
                    value_49 = T.tensor_view(
                        T.ptr_of(value_47),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_48, value_49)
                    value_50 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(vn_1, value_49, value_50, kind=BinaryKind.MUL)
                    value_51 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_50, value_51, dtype="bf16")
                    value_40 = T.tensor_view(
                        223232,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    value_52 = T.tensor_view(
                        T.ptr_of(value_40),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_51, value_52)
                    for o_m_1 in range(0, 128, 64):
                        for o_n_1 in range(0, 64, 64):
                            for o_k_1 in range(0, 64, 16):
                                value_53 = T.tensor_view(
                                    T.ptr_of(st),
                                    layout=((2, 4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (4096, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 64),
                                )
                                acc_view = T.tensor_view(
                                    T.ptr_of(value_53[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                value_54 = T.tensor_view(
                                    T.ptr_of(value_3_stages[c % 2]),
                                    layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_54[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                    layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_40_2 = T.tensor_view(
                                    223232,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 64),
                                )
                                value_55 = T.tensor_view(
                                    T.ptr_of(value_40_2),
                                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 64),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_55[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_57 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(vn_1, value_57, dtype="bf16")
                    value_58 = T.tensor_view(
                        T.ptr_of(value_56_stages[c % 3]),
                        layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_57, value_58)
                with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_6:
                    o = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(o, 0.0)
                    for o_m_2 in range(0, 64, 64):
                        for o_n_2 in range(0, 64, 64):
                            for o_k_2 in range(0, 128, 64):
                                value_60 = T.tensor_view(
                                    T.ptr_of(value_6_stages[c % 2]),
                                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 128),
                                )
                                lhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_60[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_61 = T.tensor_view(
                                    T.ptr_of(value_32_stages[c % 3]),
                                    layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                rhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_61[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o,
                                    lhs_view_5,
                                    rhs_view_5,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_60[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_61[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o,
                                    lhs_view_6,
                                    rhs_view_6,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_60[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_61[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o,
                                    lhs_view_7,
                                    rhs_view_7,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_60[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_61[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o,
                                    lhs_view_8,
                                    rhs_view_8,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_62 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1), (16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_63 = T.tensor_view(
                        T.ptr_of(value_28_stages[c % 3]),
                        layout=((64, 1), (1, 0)),
                        shape=(64, 1),
                    )
                    value_64 = T.tensor_view(
                        T.ptr_of(value_62),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1), (16, 8, 1, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_63, value_64)
                    o_1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(o),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.binary(value_65, value_64, o_1, kind=BinaryKind.MUL)
                    for o_m_3 in range(0, 64, 64):
                        for o_n_3 in range(0, 64, 64):
                            for o_k_3 in range(0, 64, 64):
                                value_66 = T.tensor_view(
                                    T.ptr_of(value_12_stages[c % 2]),
                                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 64),
                                )
                                lhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_66[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_67 = T.tensor_view(
                                    T.ptr_of(value_56_stages[c % 3]),
                                    layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 64),
                                )
                                rhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_67[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o_1,
                                    lhs_view_9,
                                    rhs_view_9,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_10 = T.tensor_view(
                                    T.ptr_of(value_66[o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_10 = T.tensor_view(
                                    T.ptr_of(value_67[o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o_1,
                                    lhs_view_10,
                                    rhs_view_10,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_11 = T.tensor_view(
                                    T.ptr_of(value_66[o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_11 = T.tensor_view(
                                    T.ptr_of(value_67[o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o_1,
                                    lhs_view_11,
                                    rhs_view_11,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                                lhs_view_12 = T.tensor_view(
                                    T.ptr_of(value_66[o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_12 = T.tensor_view(
                                    T.ptr_of(value_67[o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 64]),
                                    layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    o_1,
                                    lhs_view_12,
                                    rhs_view_12,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_68 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_69 = T.tensor_view(
                        T.ptr_of(o_1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.cast(value_69, value_68, dtype="bf16")
                    value_70 = T.tensor_view(
                        T.ptr_of(value_59_stages[c % 2]),
                        layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3),
                        shape=(64, 64),
                    )
                    T.copy(value_68, value_70)
                with scope[1:2, :32] as scope_7:
                    value_72 = T.tensor_view(
                        T.ptr_of(value_59_stages[c % 2]),
                        layout=((64, 1, 64), (64, 0, 1)) | Swizzle(3, 4, 3),
                        shape=(64, 1, 64),
                    )
                    value_71_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 4096]), layout=(64, 1, 64), shape=(64, 1, 64)
                    )
                    value_73 = T.tensor_view(
                        T.ptr_of(value_71_1), layout=(64, 1, 64), shape=(64, 1, 64)
                    )
                    T.copy_async_tensor(value_72, value_73)
                    value_74 = T.tensor_view(
                        T.ptr_of(out), layout=(1024, 48, 128), shape=(1024, 48, 128)
                    )
                    window = T.tensor_view(
                        T.ptr_of(value_74[c * 64:c * 64 + 64, cta.d1:cta.d1 + 1, cta.d0 * 64:cta.d0 * 64 + 64]),
                        layout=((64, 1, 64), (6144, 128, 1)),
                        shape=(64, 1, 64),
                    )
                    T.copy(value_73, window)
