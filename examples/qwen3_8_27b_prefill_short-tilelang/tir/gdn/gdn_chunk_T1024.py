# analysis target=nvidia.h200_sxm module=SCHED_GDN_CHUNK_T1024 function=gdn_chunk topology=cta wave=132/768
# selection requested=memory executed=memory
# memory traffic=gmem:r213.12KB/w84.78MB@logical,r159.84MB/w63.59GB@total,r213.12KB/w84.78MB@cta,r161.88KB/w84.78MB@thread;rmem:r1.47MB/w1.28MB@logical,r1.10GB/w986.17MB@total,r1.47MB/w1.28MB@cta,r26.79KB/w18.26KB@thread;smem:r322.50KB/w241.50KB@logical,r241.88MB/w181.12MB@total,r322.50KB/w241.50KB@cta,r131.65KB/w64.98KB@thread footprint=beta:16.50KB;g:33.00KB;k:768.00KB;q:768.00KB;tri:32.06KB;v:2.06MB;v319:148:16.00KB;v322:149:2.06MB;v323:150:768B;v326:151:99.00KB;v327:229:32.00KB;v330:230:4.12MB;v331:231:32.00KB;v334:232:4.12MB;v335:233:32.00KB;v338:234:4.12MB footprint-precision=exact peak=gmem:104.92MB;rmem:140.02KB;smem:112.75KB persistent=gmem:20.33MB
#   buffer=tri holds=18.29MB time=none space=cta.h,cta.c reuse=4.10MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_chunk(
    q: Tensor[(1024, 16, 128), "bf16"], k: Tensor[(1024, 16, 128), "bf16"], v: Tensor[(1024, 48, 128), "bf16"], g: Tensor[(1024, 48), "f32"], beta: Tensor[(1024, 48), "bf16"], tri: Tensor[(3, 64, 64), "f32"], scratch: Tensor[(4096,), "f32"], scratch_1: Tensor[(192,), "f32"], scratch_2: Tensor[(8192,), "f32"], scratch_3: Tensor[(8192,), "f32"], scratch_4: Tensor[(8192,), "f32"], out: Tensor[(48, 16, 28864), "f32"]
):
    with Mesh((Topology("cta", 768),), layout=(48, 16), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:1, 32:96], layout=(64,), names=("d0",)) as scope_1:
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "bf16", ((64 @ scope_1.d0, 1), (0, 0)), "rmem"]
                )
                value_2 = T.tensor_view(
                    T.ptr_of(beta[cta.d1 * 64:cta.d1 * 64 + 1, cta.d0:cta.d0 + 1]),
                    layout=((64, 1), (48, 0)),
                    shape=(64, 1),
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_1), layout=((64 @ scope_1.d0, 1), (0, 0)), shape=(64, 1)
                )
                T.copy(value_2, value_3)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "f32", ((64 @ scope_1.d0, 1), (0, 0)), "rmem"]
                )
                T.cast(value_3, value_4, dtype="f32")
                value = T.tensor_view(
                    81920, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_5 = T.tensor_view(T.ptr_of(value), layout=(64, 1), shape=(64, 1))
                T.copy(value_4, value_5)
            with Mesh((Topology("thread", 256),), layout=(32,), names=("d0",)) as scope_2:
                value_7 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "f32", ((32 @ scope_2.d0, 2, 64), (0, 64, 1)), "rmem"
                    ]
                )
                value_8 = T.tensor_view(T.ptr_of(tri), layout=(64, 64), shape=(64, 64))
                value_9 = T.tensor_view(
                    T.ptr_of(value_7),
                    layout=((32 @ scope_2.d0, 2, 64), (0, 64, 1)),
                    shape=(64, 64),
                )
                T.copy(value_8, value_9)
                value_10 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", (1, 64), "rmem"])
                value_11 = T.tensor_view(
                    T.ptr_of(g[cta.d1 * 64:cta.d1 * 64 + 1, cta.d0:cta.d0 + 1]),
                    layout=((1, 64), (0, 48)),
                    shape=(1, 64),
                )
                value_12 = T.tensor_view(T.ptr_of(value_10), layout=(1, 64), shape=(1, 64))
                T.copy(value_11, value_12)
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "f32", ((32 @ scope_2.d0, 2, 64), (0, 64, 1)), "rmem"
                    ]
                )
                T.binary(value_9, value_12, value_13, kind=BinaryKind.MUL)
                value_14 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1), "f32", ((32 @ scope_2.d0, 2, 1), (2, 1, 0)), "rmem"
                    ]
                )
                T.reduce(value_13, value_14, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                G = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1), "f32", ((32 @ scope_2.d0, 2, 1), (2, 1, 0)), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.4426950408889634)
                T.binary(value_14, scalar, G, kind=BinaryKind.MUL)
                value_6 = T.tensor_view(
                    49152, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_15 = T.tensor_view(T.ptr_of(value_6), layout=(64, 1), shape=(64, 1))
                T.copy(G, value_15)
            with Mesh((Topology("thread", 256),), layout=(64,), names=("d0",)) as scope_3:
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                value_248 = T.tensor_view(
                    81920, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_248), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_19 = T.tensor_view(
                    T.ptr_of(value_17), layout=((1, 64 @ scope_3.d0), (0, 0)), shape=(1, 64)
                )
                T.copy(value_18, value_19)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                value_6_2 = T.tensor_view(
                    49152, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_6_2), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_20), layout=((1, 64 @ scope_3.d0), (0, 0)), shape=(1, 64)
                )
                T.copy(value_21, value_22)
                eg = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.unary(value_22, eg, kind=UnaryKind.EXP2)
                value_23 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.binary(value_19, eg, value_23, kind=BinaryKind.MUL)
                value_16 = T.tensor_view(
                    82176, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)
                )
                value_24 = T.tensor_view(T.ptr_of(value_16), layout=(1, 64), shape=(1, 64))
                T.copy(value_23, value_24)
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.unary(value_22, value_25, kind=UnaryKind.NEG)
                value_26 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                value_27 = T.tensor_view(
                    T.ptr_of(value_6_2[63:63 + 1, 0:0 + 1]),
                    layout=((1, 1), (0, 0)),
                    shape=(1, 1),
                )
                value_28 = T.tensor_view(T.ptr_of(value_26), layout=(1, 1), shape=(1, 1))
                T.copy(value_27, value_28)
                value_29 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.binary(value_25, value_28, value_29, kind=BinaryKind.ADD)
                value_30 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.unary(value_29, value_30, kind=UnaryKind.EXP2)
                Fs = T.tensor_view(
                    114688, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)
                )
                value_31 = T.tensor_view(T.ptr_of(Fs), layout=(3, 64), shape=(3, 64))
                window = T.tensor_view(
                    T.ptr_of(value_31[0:0 + 1, 0:0 + 64]), layout=(1, 64), shape=(1, 64)
                )
                T.copy(value_30, window)
                value_32 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 0.08838834764831845)
                T.binary(eg, scalar_1, value_32, kind=BinaryKind.MUL)
                value_33 = T.tensor_view(T.ptr_of(Fs), layout=(3, 64), shape=(3, 64))
                window_1 = T.tensor_view(
                    T.ptr_of(value_33[1:1 + 1, 0:0 + 64]), layout=(1, 64) + 64, shape=(1, 64)
                )
                T.copy(value_32, window_1)
                value_34 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.fill(value_34, 0.0)
                value_35 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
                T.unary(value_28, value_35, kind=UnaryKind.EXP2)
                value_36 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 64), "f32", ((1, 64 @ scope_3.d0), (0, 0)), "rmem"]
                )
                T.binary(value_34, value_35, value_36, kind=BinaryKind.ADD)
                value_37 = T.tensor_view(T.ptr_of(Fs), layout=(3, 64), shape=(3, 64))
                window_2 = T.tensor_view(
                    T.ptr_of(value_37[2:2 + 1, 0:0 + 64]), layout=(1, 64) + 128, shape=(1, 64)
                )
                T.copy(value_36, window_2)
            with scope[:1, :32] as scope_4:
                value_39 = T.tensor_view(
                    T.ptr_of(k[cta.d1 * 64:cta.d1 * 64 + 1, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 1]),
                    layout=((64, 128), (2048, 1)),
                    shape=(64, 128),
                )
                value_38 = T.tensor_view(
                    32768,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_40 = T.tensor_view(
                    T.ptr_of(value_38),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_39, value_40)
                value_42 = T.tensor_view(
                    T.ptr_of(q[cta.d1 * 64:cta.d1 * 64 + 1, cta.d0 // 3:cta.d0 // 3 + 1, 0:0 + 1]),
                    layout=((64, 128), (2048, 1)),
                    shape=(64, 128),
                )
                value_41 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_43 = T.tensor_view(
                    T.ptr_of(value_41),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_42, value_43)
                value_45 = T.tensor_view(
                    T.ptr_of(v[cta.d1 * 64:cta.d1 * 64 + 1, cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((64, 128), (6144, 1)),
                    shape=(64, 128),
                )
                value_44 = T.tensor_view(
                    16384,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_46 = T.tensor_view(
                    T.ptr_of(value_44),
                    layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy_async_tensor(value_45, value_46)
            value_47 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 2, 4, 4, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        attrs=(S(3), S(0), S(2), S(5)),
                        mesh=Mesh((Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            value_50 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 2, 4, 4, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        attrs=(S(3), S(0), S(2), S(5)),
                        mesh=Mesh((Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_5:
                T.fill(value_47, 0.0)
                for o_m in range(0, 64, 64):
                    for o_n in range(0, 64, 64):
                        for o_k in range(0, 128, 64):
                            value_38_2 = T.tensor_view(
                                32768,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_48 = T.tensor_view(
                                T.ptr_of(value_38_2),
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            lhs_view = T.tensor_view(
                                T.ptr_of(value_48[o_m:o_m + 64, o_k:o_k + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_49 = T.tensor_view(
                                T.ptr_of(value_38_2),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            rhs_view = T.tensor_view(
                                T.ptr_of(value_49[o_k:o_k + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_47,
                                lhs_view,
                                rhs_view,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_1 = T.tensor_view(
                                T.ptr_of(value_48[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_1 = T.tensor_view(
                                T.ptr_of(value_49[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_47,
                                lhs_view_1,
                                rhs_view_1,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_2 = T.tensor_view(
                                T.ptr_of(value_48[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_2 = T.tensor_view(
                                T.ptr_of(value_49[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_47,
                                lhs_view_2,
                                rhs_view_2,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_3 = T.tensor_view(
                                T.ptr_of(value_48[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_3 = T.tensor_view(
                                T.ptr_of(value_49[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_47,
                                lhs_view_3,
                                rhs_view_3,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                T.fill(value_50, 0.0)
                for o_m_1 in range(0, 64, 64):
                    for o_n_1 in range(0, 64, 64):
                        for o_k_1 in range(0, 128, 64):
                            value_41_2 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_51 = T.tensor_view(
                                T.ptr_of(value_41_2),
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            lhs_view_4 = T.tensor_view(
                                T.ptr_of(value_51[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_38_3 = T.tensor_view(
                                32768,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_52 = T.tensor_view(
                                T.ptr_of(value_38_3),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            rhs_view_4 = T.tensor_view(
                                T.ptr_of(value_52[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_50,
                                lhs_view_4,
                                rhs_view_4,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_5 = T.tensor_view(
                                T.ptr_of(value_51[o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_5 = T.tensor_view(
                                T.ptr_of(value_52[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_50,
                                lhs_view_5,
                                rhs_view_5,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_6 = T.tensor_view(
                                T.ptr_of(value_51[o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_6 = T.tensor_view(
                                T.ptr_of(value_52[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_50,
                                lhs_view_6,
                                rhs_view_6,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_7 = T.tensor_view(
                                T.ptr_of(value_51[o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_7 = T.tensor_view(
                                T.ptr_of(value_52[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_50,
                                lhs_view_7,
                                rhs_view_7,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
            with Mesh(
                (Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_6:
                value_54 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_54, 0.0)
                value_55 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_253 = T.tensor_view(
                    81920, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_56 = T.tensor_view(T.ptr_of(value_253), layout=(64, 1), shape=(64, 1))
                value_57 = T.tensor_view(
                    T.ptr_of(value_55),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.copy(value_56, value_57)
                value_58 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_54, value_57, value_58, kind=BinaryKind.ADD)
                value_59 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_59, 0.0)
                value_60 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_6_3 = T.tensor_view(
                    49152, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_61 = T.tensor_view(T.ptr_of(value_6_3), layout=(64, 1), shape=(64, 1))
                value_62 = T.tensor_view(
                    T.ptr_of(value_60),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 1, 1, 1, 1), (16, 8, 1, 0, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.copy(value_61, value_62)
                value_63 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_59, value_62, value_63, kind=BinaryKind.ADD)
                value_64 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64),
                        "f32",
                        ((1, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2, 1, 1), (0, 2048, 512, 128, 64, 0, 0)),
                        "rmem",
                    ]
                )
                value_65 = T.tensor_view(
                    T.ptr_of(value_6_3), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_66 = T.tensor_view(
                    T.ptr_of(value_64),
                    layout=((1, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2, 1, 1), (0, 2048, 512, 128, 64, 0, 0)),
                    shape=(1, 64),
                )
                T.copy(value_65, value_66)
                value_67 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_63, value_66, value_67, kind=BinaryKind.SUB)
                value_68 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 0.0)
                T.binary(value_67, scalar_2, value_68, kind=BinaryKind.MIN)
                dec = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_68, dec, kind=UnaryKind.EXP2)
                value_69 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_58, dec, value_69, kind=BinaryKind.MUL)
                value_70 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_71 = T.tensor_view(
                    T.ptr_of(value_47),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_69, value_71, value_70, kind=BinaryKind.MUL)
                value_72 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_73 = T.tensor_view(
                    T.ptr_of(tri[1:1 + 1, 0:0 + 1, 0:0 + 1]), layout=(64, 64), shape=(64, 64)
                )
                value_74 = T.tensor_view(
                    T.ptr_of(value_72),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_73, value_74)
                Afr = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_70, value_74, Afr, kind=BinaryKind.MUL)
                value_53 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                value_75 = T.tensor_view(
                    T.ptr_of(value_53), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                T.copy(Afr, value_75)
                value_77 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_78 = T.tensor_view(
                    T.ptr_of(value_50),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_78, dec, value_77, kind=BinaryKind.MUL)
                value_79 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 0.08838834764831845)
                T.binary(value_77, scalar_3, value_79, kind=BinaryKind.MUL)
                value_80 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_81 = T.tensor_view(T.ptr_of(tri), layout=(64, 64), shape=(64, 64))
                value_82 = T.tensor_view(
                    T.ptr_of(value_80),
                    layout=((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_81, value_82)
                value_83 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_79, value_82, value_83, kind=BinaryKind.MUL)
                value_84 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_83, value_84, dtype="bf16")
                Pfr = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_6.d1, 2, 8 @ scope_6.d2, 2 @ scope_6.d0, 4, 4 @ scope_6.d3, 2), (16, 8, 1, 2048, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_84, Pfr, dtype="f32")
                value_76 = T.tensor_view(
                    98304,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_85 = T.tensor_view(
                    T.ptr_of(value_76),
                    layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(Pfr, value_85)
            Xs = T.tensor_view(
                49152, dtype="f32", storage="smem", layout=((64, 64), (68, 1)), shape=(64, 64)
            )
            T.fill(Xs, 0.0)
            Xb = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16), (1, 0)),
                        attrs=(S(1),),
                        mesh=Mesh((Topology("thread", 256),), layout=(16,), names=("d0",)),
                    ),
                    "rmem",
                ]
            )
            with Mesh((Topology("thread", 256),), layout=(16,), names=("d0",)) as scope_7:
                T.fill(Xb, 0.0)
                for i in range(0, 16, 1):
                    value_86 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 16 @ scope_7.d0), (0, 0)), "rmem"
                        ]
                    )
                    value_87 = T.tensor_view(
                        T.ptr_of(tri[2:2 + 1, i:i + 1, 0:0 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_88 = T.tensor_view(
                        T.ptr_of(value_86), layout=((1, 16 @ scope_7.d0), (0, 0)), shape=(1, 16)
                    )
                    T.copy(value_87, value_88)
                    value_89 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_53_2 = T.tensor_view(
                        0,
                        dtype="f32",
                        storage="smem",
                        layout=((64, 64), (68, 1)),
                        shape=(64, 64),
                    )
                    value_90 = T.tensor_view(
                        T.ptr_of(value_53_2[0 + i:0 + i + 1, 0:0 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_91 = T.tensor_view(T.ptr_of(value_89), layout=(16, 1), shape=(16, 1))
                    T.copy(value_90, value_91)
                    value_92 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_7.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_93 = T.tensor_view(
                        T.ptr_of(Xb), layout=((16, 16 @ scope_7.d0), (1, 0)), shape=(16, 16)
                    )
                    T.binary(value_93, value_91, value_92, kind=BinaryKind.MUL)
                    value_94 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_7.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_92, value_94, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_95 = T.tensor_view(
                        T.ptr_of(Xb[i:i + 1, 0:0 + 16]),
                        layout=((1, 16 @ scope_7.d0), (16, 1)),
                        shape=(1, 16),
                    )
                    T.binary(value_88, value_94, value_95, kind=BinaryKind.SUB)
                value_96 = T.tensor_view(
                    T.ptr_of(Xb), layout=((16, 16 @ scope_7.d0), (1, 0)), shape=(16, 16)
                )
                Xs_2 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_97 = T.tensor_view(
                    T.ptr_of(Xs_2), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_3 = T.tensor_view(
                    T.ptr_of(value_97[0:0 + 16, 0:0 + 16]),
                    layout=((16, 16), (68, 1)),
                    shape=(16, 16),
                )
                T.copy(value_96, window_3)
            Xb_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16), (1, 0)),
                        attrs=(S(1),),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=16,
                            outer=Layout((16,), (1,)),
                        ), names=("d0",)),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[:1, 16:32], layout=(16,), names=("d0",)) as scope_8:
                T.fill(Xb_1, 0.0)
                for i_2 in range(0, 16, 1):
                    value_98 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 16 @ scope_8.d0), (0, 0)), "rmem"
                        ]
                    )
                    value_99 = T.tensor_view(
                        T.ptr_of(tri[2:2 + 1, i_2:i_2 + 1, 0:0 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_100 = T.tensor_view(
                        T.ptr_of(value_98), layout=((1, 16 @ scope_8.d0), (0, 0)), shape=(1, 16)
                    )
                    T.copy(value_99, value_100)
                    value_101 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_53_3 = T.tensor_view(
                        0,
                        dtype="f32",
                        storage="smem",
                        layout=((64, 64), (68, 1)),
                        shape=(64, 64),
                    )
                    value_102 = T.tensor_view(
                        T.ptr_of(value_53_3[16 + i_2:16 + i_2 + 1, 16:16 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_103 = T.tensor_view(
                        T.ptr_of(value_101), layout=(16, 1), shape=(16, 1)
                    )
                    T.copy(value_102, value_103)
                    value_104 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_8.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_105 = T.tensor_view(
                        T.ptr_of(Xb_1), layout=((16, 16 @ scope_8.d0), (1, 0)), shape=(16, 16)
                    )
                    T.binary(value_105, value_103, value_104, kind=BinaryKind.MUL)
                    value_106 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_8.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_104, value_106, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_107 = T.tensor_view(
                        T.ptr_of(Xb_1[i_2:i_2 + 1, 0:0 + 16]),
                        layout=((1, 16 @ scope_8.d0), (16, 1)),
                        shape=(1, 16),
                    )
                    T.binary(value_100, value_106, value_107, kind=BinaryKind.SUB)
                value_108 = T.tensor_view(
                    T.ptr_of(Xb_1), layout=((16, 16 @ scope_8.d0), (1, 0)), shape=(16, 16)
                )
                Xs_3 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_109 = T.tensor_view(
                    T.ptr_of(Xs_3), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_4 = T.tensor_view(
                    T.ptr_of(value_109[16:16 + 16, 16:16 + 16]),
                    layout=((16, 16), (68, 1)) + 1104,
                    shape=(16, 16),
                )
                T.copy(value_108, window_4)
            Xb_2 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16), (1, 0)),
                        attrs=(S(1),),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=32,
                            outer=Layout((16,), (1,)),
                        ), names=("d0",)),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[:1, 32:48], layout=(16,), names=("d0",)) as scope_9:
                T.fill(Xb_2, 0.0)
                for i_3 in range(0, 16, 1):
                    value_110 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 16 @ scope_9.d0), (0, 0)), "rmem"
                        ]
                    )
                    value_111 = T.tensor_view(
                        T.ptr_of(tri[2:2 + 1, i_3:i_3 + 1, 0:0 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_112 = T.tensor_view(
                        T.ptr_of(value_110),
                        layout=((1, 16 @ scope_9.d0), (0, 0)),
                        shape=(1, 16),
                    )
                    T.copy(value_111, value_112)
                    value_113 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_53_4 = T.tensor_view(
                        0,
                        dtype="f32",
                        storage="smem",
                        layout=((64, 64), (68, 1)),
                        shape=(64, 64),
                    )
                    value_114 = T.tensor_view(
                        T.ptr_of(value_53_4[32 + i_3:32 + i_3 + 1, 32:32 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_115 = T.tensor_view(
                        T.ptr_of(value_113), layout=(16, 1), shape=(16, 1)
                    )
                    T.copy(value_114, value_115)
                    value_116 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_9.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_117 = T.tensor_view(
                        T.ptr_of(Xb_2), layout=((16, 16 @ scope_9.d0), (1, 0)), shape=(16, 16)
                    )
                    T.binary(value_117, value_115, value_116, kind=BinaryKind.MUL)
                    value_118 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_9.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_116, value_118, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_119 = T.tensor_view(
                        T.ptr_of(Xb_2[i_3:i_3 + 1, 0:0 + 16]),
                        layout=((1, 16 @ scope_9.d0), (16, 1)),
                        shape=(1, 16),
                    )
                    T.binary(value_112, value_118, value_119, kind=BinaryKind.SUB)
                value_120 = T.tensor_view(
                    T.ptr_of(Xb_2), layout=((16, 16 @ scope_9.d0), (1, 0)), shape=(16, 16)
                )
                Xs_4 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_121 = T.tensor_view(
                    T.ptr_of(Xs_4), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_5 = T.tensor_view(
                    T.ptr_of(value_121[32:32 + 16, 32:32 + 16]),
                    layout=((16, 16), (68, 1)) + 2208,
                    shape=(16, 16),
                )
                T.copy(value_120, window_5)
            Xb_3 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16), (1, 0)),
                        attrs=(S(1),),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=48,
                            outer=Layout((16,), (1,)),
                        ), names=("d0",)),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[:1, 48:64], layout=(16,), names=("d0",)) as scope_10:
                T.fill(Xb_3, 0.0)
                for i_4 in range(0, 16, 1):
                    value_122 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 16 @ scope_10.d0), (0, 0)), "rmem"
                        ]
                    )
                    value_123 = T.tensor_view(
                        T.ptr_of(tri[2:2 + 1, i_4:i_4 + 1, 0:0 + 1]),
                        layout=((1, 16), (0, 1)),
                        shape=(1, 16),
                    )
                    value_124 = T.tensor_view(
                        T.ptr_of(value_122),
                        layout=((1, 16 @ scope_10.d0), (0, 0)),
                        shape=(1, 16),
                    )
                    T.copy(value_123, value_124)
                    value_125 = T.alloc_tensor(
                        tensor_type=Tensor[(16, 1), "f32", (16, 1), "rmem"]
                    )
                    value_53_5 = T.tensor_view(
                        0,
                        dtype="f32",
                        storage="smem",
                        layout=((64, 64), (68, 1)),
                        shape=(64, 64),
                    )
                    value_126 = T.tensor_view(
                        T.ptr_of(value_53_5[48 + i_4:48 + i_4 + 1, 48:48 + 1]),
                        layout=((16, 1), (1, 0)),
                        shape=(16, 1),
                    )
                    value_127 = T.tensor_view(
                        T.ptr_of(value_125), layout=(16, 1), shape=(16, 1)
                    )
                    T.copy(value_126, value_127)
                    value_128 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16), "f32", ((16, 16 @ scope_10.d0), (1, 0)), "rmem"
                        ]
                    )
                    value_129 = T.tensor_view(
                        T.ptr_of(Xb_3), layout=((16, 16 @ scope_10.d0), (1, 0)), shape=(16, 16)
                    )
                    T.binary(value_129, value_127, value_128, kind=BinaryKind.MUL)
                    value_130 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (1, 16), "f32", ((1, 1, 16 @ scope_10.d0), (0, 0, 1)), "rmem"
                        ]
                    )
                    T.reduce(value_128, value_130, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                    value_131 = T.tensor_view(
                        T.ptr_of(Xb_3[i_4:i_4 + 1, 0:0 + 16]),
                        layout=((1, 16 @ scope_10.d0), (16, 1)),
                        shape=(1, 16),
                    )
                    T.binary(value_124, value_130, value_131, kind=BinaryKind.SUB)
                value_132 = T.tensor_view(
                    T.ptr_of(Xb_3), layout=((16, 16 @ scope_10.d0), (1, 0)), shape=(16, 16)
                )
                Xs_5 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_133 = T.tensor_view(
                    T.ptr_of(Xs_5), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_6 = T.tensor_view(
                    T.ptr_of(value_133[48:48 + 16, 48:48 + 16]),
                    layout=((16, 16), (68, 1)) + 3312,
                    shape=(16, 16),
                )
                T.copy(value_132, window_6)
            value_134 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16, 1), (0, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=(16, 8), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_137 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 16, 8, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=(16, 8), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_138 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16, 1), (0, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=(16, 8), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_141 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 16, 8, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=(16, 8), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 256),), layout=(16, 8), names=("d0", "d1")
            ) as scope_11:
                Xs_6 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_135 = T.tensor_view(
                    T.ptr_of(Xs_6[16:16 + 1, 16:16 + 1]),
                    layout=((16, 16, 1), (68, 1, 0)),
                    shape=(16, 16, 1),
                )
                value_136 = T.tensor_view(
                    T.ptr_of(value_134),
                    layout=((16 @ scope_11.d0, 16, 1), (0, 1, 0)),
                    shape=(16, 16, 1),
                )
                T.copy(value_135, value_136)
                value_53_6 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                value_139 = T.tensor_view(
                    T.ptr_of(value_53_6[16:16 + 1, 0:0 + 1]),
                    layout=((16, 16, 1), (68, 1, 0)),
                    shape=(16, 16, 1),
                )
                value_140 = T.tensor_view(
                    T.ptr_of(value_138),
                    layout=((16 @ scope_11.d0, 16, 1), (0, 1, 0)),
                    shape=(16, 16, 1),
                )
                T.copy(value_139, value_140)
                value_142 = T.tensor_view(
                    T.ptr_of(Xs_6), layout=((1, 16, 16), (0, 68, 1)), shape=(1, 16, 16)
                )
                value_143 = T.tensor_view(
                    T.ptr_of(value_141),
                    layout=((1, 16, 8 @ scope_11.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 16, 16),
                )
                T.copy(value_142, value_143)
                value_144 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_11.d0, 16, 8 @ scope_11.d1, 2), (256, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_140, value_143, value_144, kind=BinaryKind.MUL)
                value_145 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_11.d0, 1, 8 @ scope_11.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_144, value_145, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                Ys = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_146 = T.tensor_view(
                    T.ptr_of(Ys), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_7 = T.tensor_view(
                    T.ptr_of(value_146[16:16 + 16, 0:0 + 16]),
                    layout=((16, 16), (68, 1)) + 1088,
                    shape=(16, 16),
                )
                T.copy(value_145, window_7)
                value_147 = T.tensor_view(
                    T.ptr_of(Ys[16:16 + 1, 0:0 + 1]),
                    layout=((1, 16, 16), (0, 68, 1)),
                    shape=(1, 16, 16),
                )
                value_148 = T.tensor_view(
                    T.ptr_of(value_137),
                    layout=((1, 16, 8 @ scope_11.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 16, 16),
                )
                T.copy(value_147, value_148)
                value_149 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_11.d0, 16, 8 @ scope_11.d1, 2), (256, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_136, value_148, value_149, kind=BinaryKind.MUL)
                value_150 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_11.d0, 1, 8 @ scope_11.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_149, value_150, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                value_151 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_11.d0, 1, 8 @ scope_11.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_150, value_151, kind=UnaryKind.NEG)
                value_152 = T.tensor_view(
                    T.ptr_of(Xs_6), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_8 = T.tensor_view(
                    T.ptr_of(value_152[16:16 + 16, 0:0 + 16]),
                    layout=((16, 16), (68, 1)) + 1088,
                    shape=(16, 16),
                )
                T.copy(value_151, window_8)
            value_153 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16, 1), (0, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((16, 8), (8, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_156 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 16, 8, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((16, 8), (8, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_157 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 16, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 16, 1), (0, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((16, 8), (8, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_160 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 16, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 16, 8, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((16, 8), (8, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(16, 8), names=("d0", "d1")) as scope_12:
                Xs_7 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_154 = T.tensor_view(
                    T.ptr_of(Xs_7[48:48 + 1, 48:48 + 1]),
                    layout=((16, 16, 1), (68, 1, 0)),
                    shape=(16, 16, 1),
                )
                value_155 = T.tensor_view(
                    T.ptr_of(value_153),
                    layout=((16 @ scope_12.d0, 16, 1), (0, 1, 0)),
                    shape=(16, 16, 1),
                )
                T.copy(value_154, value_155)
                value_53_7 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                value_158 = T.tensor_view(
                    T.ptr_of(value_53_7[48:48 + 1, 32:32 + 1]),
                    layout=((16, 16, 1), (68, 1, 0)),
                    shape=(16, 16, 1),
                )
                value_159 = T.tensor_view(
                    T.ptr_of(value_157),
                    layout=((16 @ scope_12.d0, 16, 1), (0, 1, 0)),
                    shape=(16, 16, 1),
                )
                T.copy(value_158, value_159)
                value_161 = T.tensor_view(
                    T.ptr_of(Xs_7[32:32 + 1, 32:32 + 1]),
                    layout=((1, 16, 16), (0, 68, 1)),
                    shape=(1, 16, 16),
                )
                value_162 = T.tensor_view(
                    T.ptr_of(value_160),
                    layout=((1, 16, 8 @ scope_12.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 16, 16),
                )
                T.copy(value_161, value_162)
                value_163 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_12.d0, 16, 8 @ scope_12.d1, 2), (256, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_159, value_162, value_163, kind=BinaryKind.MUL)
                value_164 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_12.d0, 1, 8 @ scope_12.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_163, value_164, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                Ys_2 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_165 = T.tensor_view(
                    T.ptr_of(Ys_2), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_9 = T.tensor_view(
                    T.ptr_of(value_165[48:48 + 16, 32:32 + 16]),
                    layout=((16, 16), (68, 1)) + 3296,
                    shape=(16, 16),
                )
                T.copy(value_164, window_9)
                value_166 = T.tensor_view(
                    T.ptr_of(Ys_2[48:48 + 1, 32:32 + 1]),
                    layout=((1, 16, 16), (0, 68, 1)),
                    shape=(1, 16, 16),
                )
                value_167 = T.tensor_view(
                    T.ptr_of(value_156),
                    layout=((1, 16, 8 @ scope_12.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 16, 16),
                )
                T.copy(value_166, value_167)
                value_168 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16, 16),
                        "f32",
                        ((16 @ scope_12.d0, 16, 8 @ scope_12.d1, 2), (256, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_155, value_167, value_168, kind=BinaryKind.MUL)
                value_169 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_12.d0, 1, 8 @ scope_12.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_168, value_169, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                value_170 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 16),
                        "f32",
                        ((16 @ scope_12.d0, 1, 8 @ scope_12.d1, 2), (16, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_169, value_170, kind=UnaryKind.NEG)
                value_171 = T.tensor_view(
                    T.ptr_of(Xs_7), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_10 = T.tensor_view(
                    T.ptr_of(value_171[48:48 + 16, 32:32 + 16]),
                    layout=((16, 16), (68, 1)) + 3296,
                    shape=(16, 16),
                )
                T.copy(value_170, window_10)
            value_172 = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 32, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((16, 2, 32, 1), (0, 32, 1, 0)),
                        attrs=(B(), S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=(2, 16, 8), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_175 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32, 2, 8, 2), (0, 2, 0, 0, 1)),
                        attrs=(S(2), B(), S(3)),
                        mesh=Mesh((Topology("thread", 256),), layout=(2, 16, 8), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 256),), layout=(2, 16, 8), names=("d0", "d1", "d2")
            ) as scope_13:
                value_53_8 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                value_173 = T.tensor_view(
                    T.ptr_of(value_53_8[32:32 + 1, 0:0 + 1]),
                    layout=((32, 32, 1), (68, 1, 0)),
                    shape=(32, 32, 1),
                )
                value_174 = T.tensor_view(
                    T.ptr_of(value_172),
                    layout=((16 @ scope_13.d1, 2, 32, 1), (0, 32, 1, 0)),
                    shape=(32, 32, 1),
                )
                T.copy(value_173, value_174)
                Xs_8 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_176 = T.tensor_view(
                    T.ptr_of(Xs_8), layout=((1, 32, 32), (0, 68, 1)), shape=(1, 32, 32)
                )
                value_177 = T.tensor_view(
                    T.ptr_of(value_175),
                    layout=((1, 32, 2 @ scope_13.d0, 8 @ scope_13.d2, 2), (0, 2, 0, 0, 1)),
                    shape=(1, 32, 32),
                )
                T.copy(value_176, value_177)
                value_178 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32, 32),
                        "f32",
                        ((16 @ scope_13.d1, 2, 32, 2 @ scope_13.d0, 8 @ scope_13.d2, 2), (2048, 1024, 32, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_174, value_177, value_178, kind=BinaryKind.MUL)
                value_179 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (32, 32),
                        "f32",
                        ((16 @ scope_13.d1, 2, 1, 2 @ scope_13.d0, 8 @ scope_13.d2, 2), (64, 32, 0, 16, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_178, value_179, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                Ys_3 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_180 = T.tensor_view(
                    T.ptr_of(Ys_3), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_11 = T.tensor_view(
                    T.ptr_of(value_180[32:32 + 32, 0:0 + 32]),
                    layout=((32, 32), (68, 1)) + 2176,
                    shape=(32, 32),
                )
                T.copy(value_179, window_11)
            value_181 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 2, 32, 1), (0, 32, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_184 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32, 16, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=(8, 16), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 256),), layout=(8, 16), names=("d0", "d1")
            ) as scope_14:
                Xs_9 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_182 = T.tensor_view(
                    T.ptr_of(Xs_9[48:48 + 1, 32:32 + 1]),
                    layout=((16, 32, 1), (68, 1, 0)),
                    shape=(16, 32, 1),
                )
                value_183 = T.tensor_view(
                    T.ptr_of(value_181),
                    layout=((8 @ scope_14.d0, 2, 32, 1), (0, 32, 1, 0)),
                    shape=(16, 32, 1),
                )
                T.copy(value_182, value_183)
                Ys_4 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_185 = T.tensor_view(
                    T.ptr_of(Ys_4[32:32 + 1, 0:0 + 1]),
                    layout=((1, 32, 32), (0, 68, 1)),
                    shape=(1, 32, 32),
                )
                value_186 = T.tensor_view(
                    T.ptr_of(value_184),
                    layout=((1, 32, 16 @ scope_14.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 32, 32),
                )
                T.copy(value_185, value_186)
                value_187 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32, 32),
                        "f32",
                        ((8 @ scope_14.d0, 2, 32, 16 @ scope_14.d1, 2), (2048, 1024, 32, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_183, value_186, value_187, kind=BinaryKind.MUL)
                value_188 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32),
                        "f32",
                        ((8 @ scope_14.d0, 2, 1, 16 @ scope_14.d1, 2), (64, 32, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_187, value_188, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                value_189 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32),
                        "f32",
                        ((8 @ scope_14.d0, 2, 1, 16 @ scope_14.d1, 2), (64, 32, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_188, value_189, kind=UnaryKind.NEG)
                value_190 = T.tensor_view(
                    T.ptr_of(Xs_9), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_12 = T.tensor_view(
                    T.ptr_of(value_190[48:48 + 16, 0:0 + 32]),
                    layout=((16, 32), (68, 1)) + 3264,
                    shape=(16, 32),
                )
                T.copy(value_189, window_12)
            value_191 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 2, 32, 1), (0, 32, 1, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((8, 16), (16, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_194 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32, 16, 2), (0, 2, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((8, 16), (16, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(8, 16), names=("d0", "d1")) as scope_15:
                Xs_10 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_192 = T.tensor_view(
                    T.ptr_of(Xs_10[32:32 + 1, 32:32 + 1]),
                    layout=((16, 32, 1), (68, 1, 0)),
                    shape=(16, 32, 1),
                )
                value_193 = T.tensor_view(
                    T.ptr_of(value_191),
                    layout=((8 @ scope_15.d0, 2, 32, 1), (0, 32, 1, 0)),
                    shape=(16, 32, 1),
                )
                T.copy(value_192, value_193)
                Ys_5 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_195 = T.tensor_view(
                    T.ptr_of(Ys_5[32:32 + 1, 0:0 + 1]),
                    layout=((1, 32, 32), (0, 68, 1)),
                    shape=(1, 32, 32),
                )
                value_196 = T.tensor_view(
                    T.ptr_of(value_194),
                    layout=((1, 32, 16 @ scope_15.d1, 2), (0, 2, 0, 1)),
                    shape=(1, 32, 32),
                )
                T.copy(value_195, value_196)
                value_197 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32, 32),
                        "f32",
                        ((8 @ scope_15.d0, 2, 32, 16 @ scope_15.d1, 2), (2048, 1024, 32, 2, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_193, value_196, value_197, kind=BinaryKind.MUL)
                value_198 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32),
                        "f32",
                        ((8 @ scope_15.d0, 2, 1, 16 @ scope_15.d1, 2), (64, 32, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.reduce(value_197, value_198, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
                value_199 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 32),
                        "f32",
                        ((8 @ scope_15.d0, 2, 1, 16 @ scope_15.d1, 2), (64, 32, 0, 2, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_198, value_199, kind=UnaryKind.NEG)
                value_200 = T.tensor_view(
                    T.ptr_of(Xs_10), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                window_13 = T.tensor_view(
                    T.ptr_of(value_200[32:32 + 16, 0:0 + 32]),
                    layout=((16, 32), (68, 1)) + 2176,
                    shape=(16, 32),
                )
                T.copy(value_199, window_13)
            with Mesh(
                (Topology("thread", 256),), layout=(16, 16), names=("d0", "d1")
            ) as scope_16:
                value_202 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                Xs_11 = T.tensor_view(
                    49152,
                    dtype="f32",
                    storage="smem",
                    layout=((64, 64), (68, 1)),
                    shape=(64, 64),
                )
                value_203 = T.tensor_view(
                    T.ptr_of(Xs_11), layout=((64, 64), (68, 1)), shape=(64, 64)
                )
                value_204 = T.tensor_view(
                    T.ptr_of(value_202),
                    layout=((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                    shape=(64, 64),
                )
                T.copy(value_203, value_204)
                value_205 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "f32", ((1, 1, 16 @ scope_16.d1, 4), (0, 0, 0, 1)), "rmem"
                    ]
                )
                value_258 = T.tensor_view(
                    81920, dtype="f32", storage="smem", layout=(64, 1), shape=(64, 1)
                )
                value_206 = T.tensor_view(
                    T.ptr_of(value_258), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_207 = T.tensor_view(
                    T.ptr_of(value_205),
                    layout=((1, 1, 16 @ scope_16.d1, 4), (0, 0, 0, 1)),
                    shape=(1, 64),
                )
                T.copy(value_206, value_207)
                mb = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_204, value_207, mb, kind=BinaryKind.MUL)
                mbh = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(mb, mbh, dtype="bf16")
                value_201 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_208 = T.tensor_view(
                    T.ptr_of(value_201),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(mbh, value_208)
                value_210 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(mbh, value_210, dtype="f32")
                value_211 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(mb, value_210, value_211, kind=BinaryKind.SUB)
                value_212 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_211, value_212, dtype="bf16")
                value_209 = T.tensor_view(
                    8192,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_213 = T.tensor_view(
                    T.ptr_of(value_209),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_212, value_213)
                value_215 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 64), "f32", ((1, 1, 16 @ scope_16.d1, 4), (0, 0, 0, 1)), "rmem"
                    ]
                )
                value_16_2 = T.tensor_view(
                    82176, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)
                )
                value_216 = T.tensor_view(T.ptr_of(value_16_2), layout=(1, 64), shape=(1, 64))
                value_217 = T.tensor_view(
                    T.ptr_of(value_215),
                    layout=((1, 1, 16 @ scope_16.d1, 4), (0, 0, 0, 1)),
                    shape=(1, 64),
                )
                T.copy(value_216, value_217)
                mg = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_204, value_217, mg, kind=BinaryKind.MUL)
                mgh = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(mg, mgh, dtype="bf16")
                value_214 = T.tensor_view(
                    49152,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_218 = T.tensor_view(
                    T.ptr_of(value_214),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(mgh, value_218)
                value_220 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(mgh, value_220, dtype="f32")
                value_221 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(mg, value_220, value_221, kind=BinaryKind.SUB)
                value_222 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4, 16 @ scope_16.d0, 16 @ scope_16.d1, 4), (4, 0, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_221, value_222, dtype="bf16")
                value_219 = T.tensor_view(
                    57344,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_223 = T.tensor_view(
                    T.ptr_of(value_219),
                    layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                T.copy(value_222, value_223)
            with Mesh(
                (Topology("thread", 256),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_17:
                value_225 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 64),
                        "f32",
                        ((2, 4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (2048, 16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_225, 0.0)
                for o_m_2 in range(0, 128, 64):
                    for o_n_2 in range(0, 64, 64):
                        for o_k_2 in range(0, 64, 64):
                            acc_view = T.tensor_view(
                                T.ptr_of(value_225[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            value_44_2 = T.tensor_view(
                                16384,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_226 = T.tensor_view(
                                T.ptr_of(value_44_2),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            lhs_view_8 = T.tensor_view(
                                T.ptr_of(value_226[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_201_2 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_227 = T.tensor_view(
                                T.ptr_of(value_201_2),
                                layout=((64, 64), (1, 64)) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_8 = T.tensor_view(
                                T.ptr_of(value_227[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view,
                                lhs_view_8,
                                rhs_view_8,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_1 = T.tensor_view(
                                T.ptr_of(value_225[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_9 = T.tensor_view(
                                T.ptr_of(value_226[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_9 = T.tensor_view(
                                T.ptr_of(value_227[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_1,
                                lhs_view_9,
                                rhs_view_9,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_2 = T.tensor_view(
                                T.ptr_of(value_225[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_10 = T.tensor_view(
                                T.ptr_of(value_226[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_10 = T.tensor_view(
                                T.ptr_of(value_227[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_2,
                                lhs_view_10,
                                rhs_view_10,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_3 = T.tensor_view(
                                T.ptr_of(value_225[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_11 = T.tensor_view(
                                T.ptr_of(value_226[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_11 = T.tensor_view(
                                T.ptr_of(value_227[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_3,
                                lhs_view_11,
                                rhs_view_11,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                for o_m_3 in range(0, 128, 64):
                    for o_n_3 in range(0, 64, 64):
                        for o_k_3 in range(0, 64, 64):
                            value_228 = T.tensor_view(
                                T.ptr_of(value_225),
                                layout=((2, 4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (2048, 16, 8, 1, 4096, 512, 128, 64)),
                                shape=(128, 64),
                            )
                            acc_view_4 = T.tensor_view(
                                T.ptr_of(value_228[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            value_44_3 = T.tensor_view(
                                16384,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_229 = T.tensor_view(
                                T.ptr_of(value_44_3),
                                layout=Layout(((2, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            lhs_view_12 = T.tensor_view(
                                T.ptr_of(value_229[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_209_2 = T.tensor_view(
                                8192,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_230 = T.tensor_view(
                                T.ptr_of(value_209_2),
                                layout=((64, 64), (1, 64)) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            rhs_view_12 = T.tensor_view(
                                T.ptr_of(value_230[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_4,
                                lhs_view_12,
                                rhs_view_12,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_5 = T.tensor_view(
                                T.ptr_of(value_228[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_13 = T.tensor_view(
                                T.ptr_of(value_229[o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_13 = T.tensor_view(
                                T.ptr_of(value_230[o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_5,
                                lhs_view_13,
                                rhs_view_13,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_6 = T.tensor_view(
                                T.ptr_of(value_228[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_14 = T.tensor_view(
                                T.ptr_of(value_229[o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_14 = T.tensor_view(
                                T.ptr_of(value_230[o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_6,
                                lhs_view_14,
                                rhs_view_14,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                            acc_view_7 = T.tensor_view(
                                T.ptr_of(value_228[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_15 = T.tensor_view(
                                T.ptr_of(value_229[o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_15 = T.tensor_view(
                                T.ptr_of(value_230[o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 64]),
                                layout=(Layout((16, (2, 4, 8)), (1, (2048, 512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                acc_view_7,
                                lhs_view_15,
                                rhs_view_15,
                                atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K),
                            )
                value_231 = T.tensor_view(
                    T.ptr_of(value_225),
                    layout=((2, 4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 4, 4 @ scope_17.d3, 2), (2048, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 64),
                )
                value_224 = T.tensor_view(
                    0,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((16, 8), (2, 32)), ((256, 32), (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 64),
                )
                value_232 = T.tensor_view(
                    T.ptr_of(value_224),
                    layout=Layout(((16, 8), (2, 32)), ((256, 32), (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 64),
                )
                T.copy(value_231, value_232)
                value_234 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_234, 0.0)
                for o_m_4 in range(0, 64, 64):
                    for o_n_4 in range(0, 128, 128):
                        for o_k_4 in range(0, 64, 64):
                            value_214_2 = T.tensor_view(
                                49152,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_235 = T.tensor_view(
                                T.ptr_of(value_214_2),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_16 = T.tensor_view(
                                T.ptr_of(value_235[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_38_4 = T.tensor_view(
                                32768,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_236 = T.tensor_view(
                                T.ptr_of(value_38_4),
                                layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            rhs_view_16 = T.tensor_view(
                                T.ptr_of(value_236[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_234,
                                lhs_view_16,
                                rhs_view_16,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_17 = T.tensor_view(
                                T.ptr_of(value_235[o_m_4:o_m_4 + 64, o_k_4 + 16:o_k_4 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_17 = T.tensor_view(
                                T.ptr_of(value_236[o_k_4 + 16:o_k_4 + 16 + 16, o_n_4:o_n_4 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_234,
                                lhs_view_17,
                                rhs_view_17,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_18 = T.tensor_view(
                                T.ptr_of(value_235[o_m_4:o_m_4 + 64, o_k_4 + 32:o_k_4 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_18 = T.tensor_view(
                                T.ptr_of(value_236[o_k_4 + 32:o_k_4 + 32 + 16, o_n_4:o_n_4 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_234,
                                lhs_view_18,
                                rhs_view_18,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_19 = T.tensor_view(
                                T.ptr_of(value_235[o_m_4:o_m_4 + 64, o_k_4 + 48:o_k_4 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_19 = T.tensor_view(
                                T.ptr_of(value_236[o_k_4 + 48:o_k_4 + 48 + 16, o_n_4:o_n_4 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_234,
                                lhs_view_19,
                                rhs_view_19,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                for o_m_5 in range(0, 64, 64):
                    for o_n_5 in range(0, 128, 128):
                        for o_k_5 in range(0, 64, 64):
                            value_237 = T.tensor_view(
                                T.ptr_of(value_234),
                                layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                                shape=(64, 128),
                            )
                            value_219_2 = T.tensor_view(
                                57344,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            value_238 = T.tensor_view(
                                T.ptr_of(value_219_2),
                                layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 64),
                            )
                            lhs_view_20 = T.tensor_view(
                                T.ptr_of(value_238[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_38_5 = T.tensor_view(
                                32768,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            value_239 = T.tensor_view(
                                T.ptr_of(value_38_5),
                                layout=Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 128),
                            )
                            rhs_view_20 = T.tensor_view(
                                T.ptr_of(value_239[o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_237,
                                lhs_view_20,
                                rhs_view_20,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_21 = T.tensor_view(
                                T.ptr_of(value_238[o_m_5:o_m_5 + 64, o_k_5 + 16:o_k_5 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_21 = T.tensor_view(
                                T.ptr_of(value_239[o_k_5 + 16:o_k_5 + 16 + 16, o_n_5:o_n_5 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_237,
                                lhs_view_21,
                                rhs_view_21,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_22 = T.tensor_view(
                                T.ptr_of(value_238[o_m_5:o_m_5 + 64, o_k_5 + 32:o_k_5 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_22 = T.tensor_view(
                                T.ptr_of(value_239[o_k_5 + 32:o_k_5 + 32 + 16, o_n_5:o_n_5 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_237,
                                lhs_view_22,
                                rhs_view_22,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                            lhs_view_23 = T.tensor_view(
                                T.ptr_of(value_238[o_m_5:o_m_5 + 64, o_k_5 + 48:o_k_5 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_23 = T.tensor_view(
                                T.ptr_of(value_239[o_k_5 + 48:o_k_5 + 48 + 16, o_n_5:o_n_5 + 128]),
                                layout=(Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 128),
                            )
                            T.tiled_mma(
                                value_237,
                                lhs_view_23,
                                rhs_view_23,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                wh = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "bf16",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_240 = T.tensor_view(
                    T.ptr_of(value_234),
                    layout=((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                    shape=(64, 128),
                )
                T.cast(value_240, wh, dtype="bf16")
                value_241 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(wh, value_241, dtype="f32")
                value_233 = T.tensor_view(
                    32768,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_242 = T.tensor_view(
                    T.ptr_of(value_233),
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy(value_241, value_242)
                value_244 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(wh, value_244, dtype="f32")
                value_245 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_240, value_244, value_245, kind=BinaryKind.SUB)
                wlo = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "bf16",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_245, wlo, dtype="bf16")
                value_246 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 128),
                        "f32",
                        ((4 @ scope_17.d1, 2, 8 @ scope_17.d2, 2 @ scope_17.d0, 8, 4 @ scope_17.d3, 2), (16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(wlo, value_246, dtype="f32")
                value_243 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_247 = T.tensor_view(
                    T.ptr_of(value_243),
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                T.copy(value_246, value_247)
            T.fill(out, 0.0)
            with scope[:1, :32] as scope_18:
                value_76_2 = T.tensor_view(
                    98304,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_249 = T.tensor_view(
                    T.ptr_of(value_76_2),
                    layout=Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 64),
                )
                value_248_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 4096]), layout=(64, 64), shape=(64, 64)
                )
                value_250 = T.tensor_view(
                    T.ptr_of(value_248_1), layout=(64, 64), shape=(64, 64)
                )
                T.copy_async_tensor(value_249, value_250)
                value_251 = T.tensor_view(
                    T.ptr_of(value_248_1), layout=((1, 1, 4096), (0, 0, 1)), shape=(1, 1, 4096)
                )
                value_252 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 16, 28864), shape=(48, 16, 28864)
                )
                window_14 = T.tensor_view(
                    T.ptr_of(value_252[cta.d0:cta.d0 + 1, cta.d1:cta.d1 + 1, 24576:24576 + 4096]),
                    layout=((1, 1, 4096), (461824, 28864, 1)),
                    shape=(1, 1, 4096),
                )
                T.copy(value_251, window_14)
                Fs_2 = T.tensor_view(
                    114688, dtype="f32", storage="smem", layout=(3, 64), shape=(3, 64)
                )
                value_254 = T.tensor_view(T.ptr_of(Fs_2), layout=(3, 64), shape=(3, 64))
                value_253_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 192]), layout=(3, 64), shape=(3, 64)
                )
                value_255 = T.tensor_view(T.ptr_of(value_253_1), layout=(3, 64), shape=(3, 64))
                T.copy_async_tensor(value_254, value_255)
                value_256 = T.tensor_view(
                    T.ptr_of(value_253_1), layout=((1, 1, 192), (0, 0, 1)), shape=(1, 1, 192)
                )
                value_257 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 16, 28864), shape=(48, 16, 28864)
                )
                window_15 = T.tensor_view(
                    T.ptr_of(value_257[cta.d0:cta.d0 + 1, cta.d1:cta.d1 + 1, 28672:28672 + 192]),
                    layout=((1, 1, 192), (461824, 28864, 1)),
                    shape=(1, 1, 192),
                )
                T.copy(value_256, window_15)
            with scope[:1, :32] as scope_19:
                value_224_2 = T.tensor_view(
                    0,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((16, 8), (2, 32)), ((256, 32), (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 64),
                )
                value_259 = T.tensor_view(
                    T.ptr_of(value_224_2),
                    layout=Layout(((16, 8), (2, 32)), ((256, 32), (4096, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 64),
                )
                value_258_1 = T.tensor_view(
                    T.ptr_of(scratch_2[0:0 + 8192]), layout=(128, 64), shape=(128, 64)
                )
                value_260 = T.tensor_view(
                    T.ptr_of(value_258_1), layout=(128, 64), shape=(128, 64)
                )
                T.copy_async_tensor(value_259, value_260)
                value_261 = T.tensor_view(
                    T.ptr_of(value_258_1), layout=((1, 1, 8192), (0, 0, 1)), shape=(1, 1, 8192)
                )
                value_262 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 16, 28864), shape=(48, 16, 28864)
                )
                window_16 = T.tensor_view(
                    T.ptr_of(value_262[cta.d0:cta.d0 + 1, cta.d1:cta.d1 + 1, 0:0 + 8192]),
                    layout=((1, 1, 8192), (461824, 28864, 1)),
                    shape=(1, 1, 8192),
                )
                T.copy(value_261, window_16)
                value_233_2 = T.tensor_view(
                    32768,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_264 = T.tensor_view(
                    T.ptr_of(value_233_2),
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_263_1 = T.tensor_view(
                    T.ptr_of(scratch_3[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
                )
                value_265 = T.tensor_view(
                    T.ptr_of(value_263_1), layout=(64, 128), shape=(64, 128)
                )
                T.copy_async_tensor(value_264, value_265)
                value_266 = T.tensor_view(
                    T.ptr_of(value_263_1), layout=((1, 1, 8192), (0, 0, 1)), shape=(1, 1, 8192)
                )
                value_267 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 16, 28864), shape=(48, 16, 28864)
                )
                window_17 = T.tensor_view(
                    T.ptr_of(value_267[cta.d0:cta.d0 + 1, cta.d1:cta.d1 + 1, 8192:8192 + 8192]),
                    layout=((1, 1, 8192), (461824, 28864, 1)),
                    shape=(1, 1, 8192),
                )
                T.copy(value_266, window_17)
                value_243_2 = T.tensor_view(
                    65536,
                    dtype="f32",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_269 = T.tensor_view(
                    T.ptr_of(value_243_2),
                    layout=Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 128),
                )
                value_268_1 = T.tensor_view(
                    T.ptr_of(scratch_4[0:0 + 8192]), layout=(64, 128), shape=(64, 128)
                )
                value_270 = T.tensor_view(
                    T.ptr_of(value_268_1), layout=(64, 128), shape=(64, 128)
                )
                T.copy_async_tensor(value_269, value_270)
                value_271 = T.tensor_view(
                    T.ptr_of(value_268_1), layout=((1, 1, 8192), (0, 0, 1)), shape=(1, 1, 8192)
                )
                value_272 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 16, 28864), shape=(48, 16, 28864)
                )
                window_18 = T.tensor_view(
                    T.ptr_of(value_272[cta.d0:cta.d0 + 1, cta.d1:cta.d1 + 1, 16384:16384 + 8192]),
                    layout=((1, 1, 8192), (461824, 28864, 1)),
                    shape=(1, 1, 8192),
                )
                T.copy(value_271, window_18)
