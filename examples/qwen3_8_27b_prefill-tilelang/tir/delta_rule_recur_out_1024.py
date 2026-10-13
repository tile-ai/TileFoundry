# analysis target=nvidia.h200_sxm module=DeltaRuleRecur function=recur_out topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r1.13MB/w12.25MB@logical,r108.38MB/w1.15GB@total,r1.13MB/w12.25MB@cta,r1.13MB/w12.25MB@thread;rmem:r4.78MB/w3.59MB@logical,r459.33MB/w389.64MB@total,r4.78MB/w4.06MB@cta,r42.16KB/w33.02KB@thread;smem:r1.89MB/w1.64MB@logical,r181.14MB/w157.12MB@total,r1.89MB/w1.64MB@cta,r349.44KB/w1.01MB@thread footprint=gc:256B;k:16.00KB;q:16.00KB;v77:327:16.00KB;v78:328:8.00KB footprint-precision=lower_bound peak=gmem:50.20MB;rmem:64.27KB;smem:226.00KB persistent=gmem:38.19MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def recur_out(
    q: Tensor[(1024, 16, 128), "bf16"], k: Tensor[(1024, 16, 128), "bf16"], wup: Tensor[(1024, 48, 320), "bf16"], gc: Tensor[(48, 1024), "f32"], scratch: Tensor[(4096,), "bf16"], out: Tensor[(1024, 48, 128), "bf16"]
):
    with Mesh((Topology("cta", 96),), layout=(2, 48), names=("d0", "d1")) as cta:
        st = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 64),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        T.fill(out, 0.0)
        value = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        value_1 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        gl = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
        g64 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(64,), "f32", (64,), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 64),
                "bf16",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        st_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 64),
                "f32",
                ((2, 8, 2, 4, 2, 4, 8), (4096, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        el = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
        value_5 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "f32", (1, 1), "rmem"])
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        vn = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        uu = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        vn_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        edr = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1), (1, 8, 16, 0)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        o = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        o_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", ((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        eqr = T.alloc_tensor(
            tensor_type=Tensor[(64, 1), "f32", ((8, 2, 4, 1), (1, 8, 16, 0)), "rmem"]
        )
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                T.fill(st, 0.0)
            ws_stages = (
                T.tensor_view(49152, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            ks_stages = (
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            qs_stages = (
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 128)),
            )
            us_stages = (
                T.tensor_view(172032, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            ps_stages = (
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(196608, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            gs_stages = (
                T.tensor_view(222720, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
                T.tensor_view(222976, dtype="f32", storage="smem", layout=(1, 64), shape=(1, 64)),
            )
            eds_stages = (
                T.tensor_view(221184, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221440, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(221696, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            eqs_stages = (
                T.tensor_view(221952, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222208, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
                T.tensor_view(222464, dtype="f32", storage="smem", layout=(64,), shape=(64,)),
            )
            sb_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            vd = T.tensor_view(
                223232,
                dtype="bf16",
                storage="smem",
                layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                shape=(64, 64),
            )
            vnb_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(155648, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            os_stages = (
                T.tensor_view(204800, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
                T.tensor_view(212992, dtype="bf16", storage="smem", layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3), shape=(64, 64)),
            )
            tile_21_1 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 4096]), layout=(64, 1, 64), shape=(64, 1, 64)
            )
            for c in range(0, 16, 1):
                with scope[2:, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(wup[c * 64:c * 64 + 64, cta.d1:cta.d1 + 1, 0:0 + 128]),
                        layout=((64, 1, 128), (15360, 320, 1)),
                        shape=(64, 1, 128),
                    )
                    tile_1 = T.tensor_view(
                        T.ptr_of(tile[0:0 + 64, 0:0 + 1, 0:0 + 128]),
                        layout=((64, 128), (15360, 1)),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile_1, ws_stages[c % 2])
                    tile_2 = T.tensor_view(
                        T.ptr_of(k[c * 64:c * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 128]),
                        layout=((64, 1, 128), (2048, 128, 1)),
                        shape=(64, 1, 128),
                    )
                    tile_3 = T.tensor_view(
                        T.ptr_of(tile_2[0:0 + 64, 0:0 + 1, 0:0 + 128]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_3, ks_stages[c % 2])
                    tile_4 = T.tensor_view(
                        T.ptr_of(q[c * 64:c * 64 + 64, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 128]),
                        layout=((64, 1, 128), (2048, 128, 1)),
                        shape=(64, 1, 128),
                    )
                    tile_5 = T.tensor_view(
                        T.ptr_of(tile_4[0:0 + 64, 0:0 + 1, 0:0 + 128]),
                        layout=((64, 128), (2048, 1)),
                        shape=(64, 128),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_3:
                        T.copy_async_tensor(tile_5, qs_stages[c % 2])
                    tile_6 = T.tensor_view(
                        T.ptr_of(wup[c * 64:c * 64 + 64, cta.d1:cta.d1 + 1, 128 + (cta.d0 * 64):128 + (cta.d0 * 64) + 64]),
                        layout=((64, 1, 64), (15360, 320, 1)),
                        shape=(64, 1, 64),
                    )
                    tile_7 = T.tensor_view(
                        T.ptr_of(tile_6[0:0 + 64, 0:0 + 1, 0:0 + 64]),
                        layout=((64, 64), (15360, 1)),
                        shape=(64, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_4:
                        T.copy_async_tensor(tile_7, us_stages[c % 2])
                    tile_8 = T.tensor_view(
                        T.ptr_of(wup[c * 64:c * 64 + 64, cta.d1:cta.d1 + 1, 256:256 + 64]),
                        layout=((64, 1, 64), (15360, 320, 1)),
                        shape=(64, 1, 64),
                    )
                    tile_9 = T.tensor_view(
                        T.ptr_of(tile_8[0:0 + 64, 0:0 + 1, 0:0 + 64]),
                        layout=((64, 64), (15360, 1)),
                        shape=(64, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_5:
                        T.copy_async_tensor(tile_9, ps_stages[c % 2])
                    tile_10 = T.tensor_view(
                        T.ptr_of(gc[cta.d1:cta.d1 + 1, c * 64:c * 64 + 64]),
                        layout=((1, 64), (1024, 1)),
                        shape=(1, 64),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_6:
                        T.copy_async_tensor(tile_10, gs_stages[c % 2])
                with Mesh(scope[:1, :64], layout=(64,), names=("d0",)) as scope_2:
                    tile_11 = T.tensor_view(
                        T.ptr_of(gs_stages[c % 2][0:0 + 1, 63:63 + 1]),
                        layout=((1, 1), (64, 1)) + 63,
                        shape=(1, 1),
                    )
                    tile_12 = T.tensor_view(
                        T.ptr_of(tile_11[0:0 + 1, 0:0 + 1]), layout=(1,) + 63, shape=(1,)
                    )
                    with scope_2 as threads_7:
                        T.copy(tile_12, gl)
                    tile_13 = T.tensor_view(
                        T.ptr_of(gs_stages[c % 2][0:0 + 1, 0:0 + 64]), layout=(64,), shape=(64,)
                    )
                    with scope_2 as threads_8:
                        dst_frame = T.tensor_view(
                            T.ptr_of(g64[0:0 + 64]), layout=(64 @ threads_8.d0,), shape=(64,)
                        )
                        T.copy(tile_13, dst_frame)
                        rhs_frame = T.tensor_view(
                            T.ptr_of(g64[0:0 + 64]), layout=(64 @ threads_8.d0,), shape=(64,)
                        )
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 64]),
                            layout=(64 @ threads_8.d0,),
                            shape=(64,),
                        )
                        T.binary(gl, rhs_frame, dst_frame_1, kind=BinaryKind.SUB)
                        src_frame = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 64]),
                            layout=(64 @ threads_8.d0,),
                            shape=(64,),
                        )
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(value[0:0 + 64]), layout=(64 @ threads_8.d0,), shape=(64,)
                        )
                        T.unary(src_frame, dst_frame_2, kind=UnaryKind.EXP)
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(value[0:0 + 64]), layout=(64 @ threads_8.d0,), shape=(64,)
                        )
                        T.copy(src_frame_1, eds_stages[c % 3])
                        src_frame_2 = T.tensor_view(
                            T.ptr_of(g64[0:0 + 64]), layout=(64 @ threads_8.d0,), shape=(64,)
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 64]),
                            layout=(64 @ threads_8.d0,),
                            shape=(64,),
                        )
                        T.unary(src_frame_2, dst_frame_3, kind=UnaryKind.EXP)
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 0.08838834764831845)
                    with scope_2 as threads_13:
                        lhs_frame = T.tensor_view(
                            T.ptr_of(value_3[0:0 + 64]),
                            layout=(64 @ threads_13.d0,),
                            shape=(64,),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 64]),
                            layout=(64 @ threads_13.d0,),
                            shape=(64,),
                        )
                        T.binary(lhs_frame, scalar, dst_frame_4, kind=BinaryKind.MUL)
                        src_frame_3 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 64]),
                            layout=(64 @ threads_13.d0,),
                            shape=(64,),
                        )
                        T.copy(src_frame_3, eqs_stages[c % 3])
                with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                    with scope_3 as threads_15:
                        src_frame_4 = T.tensor_view(
                            T.ptr_of(st[0:0 + 128, 0:0 + 64]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 64),
                        )
                        dst_frame_5 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 128, 0:0 + 64]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 64),
                        )
                        T.cast(src_frame_4, dst_frame_5, dtype="bf16")
                        src_frame_5 = T.tensor_view(
                            T.ptr_of(value_4[0:0 + 128, 0:0 + 64]),
                            layout=((2, 8 @ threads_15.d1, 2, 4 @ threads_15.d0, 2, 4 @ threads_15.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 64),
                        )
                        T.copy(src_frame_5, sb_stages[c % 3])
                    tile_14 = T.tensor_view(
                        T.ptr_of(gs_stages[c % 2][0:0 + 1, 63:63 + 1]),
                        layout=((1, 1), (64, 1)) + 63,
                        shape=(1, 1),
                    )
                    tile_15 = T.tensor_view(
                        T.ptr_of(tile_14[0:0 + 1, 0:0 + 1]), layout=(1, 1) + 63, shape=(1, 1)
                    )
                    with scope_3 as threads_17:
                        T.copy(tile_15, value_5)
                        T.unary(value_5, el, kind=UnaryKind.EXP)
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(st[0:0 + 128, 0:0 + 64]),
                            layout=((2, 8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 64),
                        )
                        dst_frame_6 = T.tensor_view(
                            T.ptr_of(st_1[0:0 + 128, 0:0 + 64]),
                            layout=((2, 8 @ threads_17.d1, 2, 4 @ threads_17.d0, 2, 4 @ threads_17.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 64),
                        )
                        T.binary(lhs_frame_1, el, dst_frame_6, kind=BinaryKind.MUL)
                    tile_16 = T.tensor_view(
                        T.ptr_of(ks_stages[c % 2][0:0 + 64, 0:0 + 128]),
                        layout=Layout(((2, 1, 64), (4, 2, 8)), ((4096, 1024, 1), (1024, 512, 64))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_3 as threads_20:
                        dst_frame_7 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(us_stages[c % 2], dst_frame_7)
                        src_frame_6 = T.tensor_view(
                            T.ptr_of(value_8[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_8 = T.tensor_view(
                            T.ptr_of(uu[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_20.d1, 2, 4 @ threads_20.d0, 2, 4 @ threads_20.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_6, dst_frame_8, dtype="f32")
                        T.fill(vn_1, 0.0)
                    tile_17 = T.tensor_view(
                        T.ptr_of(sb_stages[c % 3][0:0 + 128, 0:0 + 64]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_3 as threads_23:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 64, 64):
                                for o_k in range(0, 128, 64):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(vn_1[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(ws_stages[c % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(tile_17[o_k:o_k + 16, o_n:o_n + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_23),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(vn_1[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(ws_stages[c % 2][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_23),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(vn_1[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(ws_stages[c % 2][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_23),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(vn_1[o_m:o_m + 64, o_n:o_n + 64]),
                                        layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(ws_stages[c % 2][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(tile_17[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_23),
                                    )
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(uu[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(vn_1[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_9 = T.tensor_view(
                            T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_23.d1, 2, 4 @ threads_23.d0, 2, 4 @ threads_23.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_2, rhs_frame_1, dst_frame_9, kind=BinaryKind.SUB)
                    tile_18 = T.tensor_view(
                        T.ptr_of(eds_stages[c % 3][0:0 + 64]), layout=(64, 1), shape=(64, 1)
                    )
                    with scope_3 as threads_25:
                        dst_frame_10 = T.tensor_view(
                            T.ptr_of(edr[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1), (1, 8, 16, 0)),
                            shape=(64, 1),
                        )
                        T.copy(tile_18, dst_frame_10)
                        lhs_frame_3 = T.tensor_view(
                            T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(edr[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 1), (1, 8, 16, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_11 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_3, rhs_frame_2, dst_frame_11, kind=BinaryKind.MUL)
                        src_frame_7 = T.tensor_view(
                            T.ptr_of(value_7[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_12 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_7, dst_frame_12, dtype="bf16")
                        src_frame_8 = T.tensor_view(
                            T.ptr_of(value_6[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_8, vd)
                        for o_m_1 in range(0, 128, 64):
                            for o_n_1 in range(0, 64, 64):
                                for o_k_1 in range(0, 64, 16):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(st_1[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                        layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(tile_16[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((1, 64), (2, 8)), ((1024, 1), (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(vd[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN, mesh=threads_25),
                                    )
                        src_frame_9 = T.tensor_view(
                            T.ptr_of(vn[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_13 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_9, dst_frame_13, dtype="bf16")
                        src_frame_10 = T.tensor_view(
                            T.ptr_of(value_9[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_25.d1, 2, 4 @ threads_25.d0, 2, 4 @ threads_25.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_10, vnb_stages[c % 3])
                with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    with scope_5 as threads_32:
                        T.fill(o_1, 0.0)
                    tile_19 = T.tensor_view(
                        T.ptr_of(sb_stages[c % 3][0:0 + 128, 0:0 + 64]),
                        layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    with scope_5 as threads_33:
                        for o_m_2 in range(0, 64, 64):
                            for o_n_2 in range(0, 64, 64):
                                for o_k_2 in range(0, 128, 64):
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(qs_stages[c % 2][o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(tile_19[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_33),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(qs_stages[c % 2][o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(tile_19[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_33),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(qs_stages[c % 2][o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(tile_19[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_33),
                                    )
                                    acc_view_8 = T.tensor_view(
                                        T.ptr_of(o_1[o_m_2:o_m_2 + 64, o_n_2:o_n_2 + 64]),
                                        layout=((8 @ threads_33.d1, 2, 4 @ threads_33.d0, 2, 4 @ threads_33.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_8 = T.tensor_view(
                                        T.ptr_of(qs_stages[c % 2][o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_8 = T.tensor_view(
                                        T.ptr_of(tile_19[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_8,
                                        lhs_view_8,
                                        rhs_view_8,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_33),
                                    )
                    tile_20 = T.tensor_view(
                        T.ptr_of(eqs_stages[c % 3][0:0 + 64]), layout=(64, 1), shape=(64, 1)
                    )
                    with scope_5 as threads_34:
                        dst_frame_14 = T.tensor_view(
                            T.ptr_of(eqr[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 1), (1, 8, 16, 0)),
                            shape=(64, 1),
                        )
                        T.copy(tile_20, dst_frame_14)
                        lhs_frame_4 = T.tensor_view(
                            T.ptr_of(o_1[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        rhs_frame_3 = T.tensor_view(
                            T.ptr_of(eqr[0:0 + 64, 0:0 + 1]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 1), (1, 8, 16, 0)),
                            shape=(64, 1),
                        )
                        dst_frame_15 = T.tensor_view(
                            T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.binary(lhs_frame_4, rhs_frame_3, dst_frame_15, kind=BinaryKind.MUL)
                        for o_m_3 in range(0, 64, 64):
                            for o_n_3 in range(0, 64, 64):
                                for o_k_3 in range(0, 64, 64):
                                    acc_view_9 = T.tensor_view(
                                        T.ptr_of(o[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                        layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_9 = T.tensor_view(
                                        T.ptr_of(ps_stages[c % 2][o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_9 = T.tensor_view(
                                        T.ptr_of(vnb_stages[c % 3][o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_9,
                                        lhs_view_9,
                                        rhs_view_9,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_34),
                                    )
                                    acc_view_10 = T.tensor_view(
                                        T.ptr_of(o[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                        layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_10 = T.tensor_view(
                                        T.ptr_of(ps_stages[c % 2][o_m_3:o_m_3 + 64, o_k_3 + 16:o_k_3 + 16 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_10 = T.tensor_view(
                                        T.ptr_of(vnb_stages[c % 3][o_k_3 + 16:o_k_3 + 16 + 16, o_n_3:o_n_3 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_10,
                                        lhs_view_10,
                                        rhs_view_10,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_34),
                                    )
                                    acc_view_11 = T.tensor_view(
                                        T.ptr_of(o[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                        layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_11 = T.tensor_view(
                                        T.ptr_of(ps_stages[c % 2][o_m_3:o_m_3 + 64, o_k_3 + 32:o_k_3 + 32 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_11 = T.tensor_view(
                                        T.ptr_of(vnb_stages[c % 3][o_k_3 + 32:o_k_3 + 32 + 16, o_n_3:o_n_3 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_11,
                                        lhs_view_11,
                                        rhs_view_11,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_34),
                                    )
                                    acc_view_12 = T.tensor_view(
                                        T.ptr_of(o[o_m_3:o_m_3 + 64, o_n_3:o_n_3 + 64]),
                                        layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 64),
                                    )
                                    lhs_view_12 = T.tensor_view(
                                        T.ptr_of(ps_stages[c % 2][o_m_3:o_m_3 + 64, o_k_3 + 48:o_k_3 + 48 + 16]),
                                        layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_view_12 = T.tensor_view(
                                        T.ptr_of(vnb_stages[c % 3][o_k_3 + 48:o_k_3 + 48 + 16, o_n_3:o_n_3 + 64]),
                                        layout=(Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 64),
                                    )
                                    T.tiled_mma(
                                        acc_view_12,
                                        lhs_view_12,
                                        rhs_view_12,
                                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_34),
                                    )
                        src_frame_11 = T.tensor_view(
                            T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        dst_frame_16 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.cast(src_frame_11, dst_frame_16, dtype="bf16")
                        src_frame_12 = T.tensor_view(
                            T.ptr_of(value_10[0:0 + 64, 0:0 + 64]),
                            layout=((8 @ threads_34.d1, 2, 4 @ threads_34.d0, 2, 4 @ threads_34.d2, 8), (1, 8, 16, 64, 128, 512)),
                            shape=(64, 64),
                        )
                        T.copy(src_frame_12, os_stages[c % 2])
                with scope[1:2, :32] as scope_6:
                    tile_22 = T.tensor_view(
                        T.ptr_of(os_stages[c % 2][0:0 + 64, 0:0 + 64]),
                        layout=(64, 1, 64) | Swizzle(3, 4, 3),
                        shape=(64, 1, 64),
                    )
                    with Mesh(scope_6, layout=(32,), names=("d0",)) as threads_39:
                        T.copy_async_tensor(tile_22, tile_21_1)
                    window = T.tensor_view(
                        T.ptr_of(out[c * 64:c * 64 + 64, cta.d1:cta.d1 + 1, cta.d0 * 64:cta.d0 * 64 + 64]),
                        layout=((64, 1, 64), (6144, 128, 1)),
                        shape=(64, 1, 64),
                    )
                    T.copy(tile_21_1, window)
                with Mesh(scope[:1], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads_40:
                    st_1_view = T.tensor_view(
                        T.ptr_of(st_1[0:0 + 128, 0:0 + 64]),
                        layout=((2, 8 @ threads_40.d1, 2, 4 @ threads_40.d0, 2, 4 @ threads_40.d2, 8), (4096, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 64),
                    )
                    T.copy(st_1_view, st)
