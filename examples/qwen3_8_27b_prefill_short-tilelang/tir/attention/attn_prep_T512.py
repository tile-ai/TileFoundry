# analysis target=nvidia.h200_sxm module=SCHED_ATTN_PREP_T512 function=attn_prep topology=cta wave=132/512
# selection requested=memory executed=memory
# memory traffic=gmem:r33.50KB/w8.03MB@logical,r16.75MB/w4.02GB@total,r33.50KB/w8.03MB@cta,r16.09KB/w8.03MB@thread;rmem:r271.79KB/w208.33KB@logical,r135.89MB/w104.16MB@total,r271.79KB/w208.33KB@cta,r1.29KB/w704B@thread;smem:r14.00KB/w14.00KB@logical,r7.00MB/w7.00MB@total,r14.00KB/w14.00KB@cta,r10.51KB/w32B@thread footprint=gamma_k:512B;gamma_q:512B;v124:82:256B;v144:83:256B;v149:84:1.50KB;v158:88:2.00KB;v50:60:1.50KB;v70:61:1.50KB;v75:62:9.00KB footprint-precision=lower_bound peak=gmem:30.14MB;rmem:62.60KB;smem:12.00KB persistent=gmem:14.13MB
#   buffer=gamma_q holds=17.00KB time=none space=cta.t reuse=65.50KB fits=yes precision=lower_bound
#   buffer=gamma_k holds=17.00KB time=none space=cta.t reuse=65.50KB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_prep(
    qkv: Tensor[(512, 14336), "bf16"], gamma_q: Tensor[(256,), "bf16"], gamma_k: Tensor[(256,), "bf16"], cos: Tensor[(512, 64), "bf16"], sin: Tensor[(512, 64), "bf16"], scratch: Tensor[(768,), "bf16"], scratch_1: Tensor[(768,), "bf16"], scratch_2: Tensor[(4608,), "bf16"], scratch_3: Tensor[(128,), "bf16"], scratch_4: Tensor[(128,), "bf16"], scratch_5: Tensor[(768,), "bf16"], scratch_6: Tensor[(1024,), "bf16"], out: Tensor[(512, 32, 256), "bf16"]
):
    with Mesh((Topology("cta", 512),), layout=(512,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 768]), layout=(1, 24, 32), shape=(1, 24, 32)
        )
        r_lo = T.alloc_tensor(tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"])
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        lo = T.alloc_tensor(tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"])
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        value_7 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(24, 256), shape=(24, 256)
        )
        n = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "bf16", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "f32", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "f32", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        xf = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "f32", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "bf16", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(24, 1), "f32", ((24, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(24, 1), "f32", ((24, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        ms = T.alloc_tensor(
            tensor_type=Tensor[(24, 1), "f32", ((24, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(24, 256), "f32", ((24, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "f32", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "f32", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "bf16", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        c_lo = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        hi = T.alloc_tensor(tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"])
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        s_lo = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_24_1 = T.tensor_view(
            T.ptr_of(scratch_1[0:0 + 768]), layout=(1, 24, 32), shape=(1, 24, 32)
        )
        r_hi = T.alloc_tensor(tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"])
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        c_hi = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_30 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "bf16", ((24, 32), (0, 0)), "rmem"]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[(24, 32), "f32", ((24, 32), (0, 0)), "rmem"]
        )
        s_hi = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_33 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_34_1 = T.tensor_view(
            T.ptr_of(scratch_2[0:0 + 4608]), layout=(1, 24, 192), shape=(1, 24, 192)
        )
        value_35_1 = T.tensor_view(
            T.ptr_of(scratch_3[0:0 + 128]), layout=(1, 4, 32), shape=(1, 4, 32)
        )
        r_lo_1 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"])
        value_36 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_37 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_38 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        value_39 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        lo_1 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_40 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        value_41 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(4, 256), shape=(4, 256)
        )
        n_1 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "bf16", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_42 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "f32", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_43 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "f32", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        xf_1 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "f32", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_44 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "bf16", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_45 = T.alloc_tensor(
            tensor_type=Tensor[(4, 1), "f32", ((4, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        value_46 = T.alloc_tensor(
            tensor_type=Tensor[(4, 1), "f32", ((4, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        ms_1 = T.alloc_tensor(
            tensor_type=Tensor[(4, 1), "f32", ((4, 1, 1, 1), (1, 0, 0, 0)), "rmem"]
        )
        value_47 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "f32", ((4, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_48 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "f32", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_49 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "f32", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_50 = T.alloc_tensor(
            tensor_type=Tensor[(1, 256), "bf16", ((1, 2, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        c_lo_1 = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_51 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_52 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_53 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        value_54 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_55 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        hi_1 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_56 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        s_lo_1 = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_57 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_58_1 = T.tensor_view(
            T.ptr_of(scratch_4[0:0 + 128]), layout=(1, 4, 32), shape=(1, 4, 32)
        )
        r_hi_1 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"])
        value_59 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_60 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_61 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        value_62 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        c_hi_1 = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_63 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_64 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        value_65 = T.alloc_tensor(
            tensor_type=Tensor[(4, 32), "bf16", ((4, 32), (0, 0)), "rmem"]
        )
        value_66 = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4, 32), (0, 0)), "rmem"])
        s_hi_1 = T.alloc_tensor(tensor_type=Tensor[(1, 32), "f32", ((1, 32), (0, 0)), "rmem"])
        value_67 = T.alloc_tensor(
            tensor_type=Tensor[(1, 32), "bf16", ((1, 32), (0, 0)), "rmem"]
        )
        value_68_1 = T.tensor_view(
            T.ptr_of(scratch_5[0:0 + 768]), layout=(1, 4, 192), shape=(1, 4, 192)
        )
        value_69_1 = T.tensor_view(
            T.ptr_of(scratch_6[0:0 + 1024]), layout=(1, 4, 256), shape=(1, 4, 256)
        )
        value_70 = T.alloc_tensor(
            tensor_type=Tensor[(4, 256), "bf16", ((4, 32, 8), (0, 0, 1)), "rmem"]
        )
        with Mesh((Topology("thread", 1024),), layout=(32, 32), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(qkv[cta.d0:cta.d0 + 1, 0:0 + 12288]),
                layout=((1, 12288), (14336, 1)),
                shape=(1, 12288),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[0:0 + 1, 0:0 + 12288]), layout=(24, 512), shape=(24, 512)
            )
            tile_2 = T.tensor_view(
                T.ptr_of(tile_1[0:0 + 24, 0:0 + 256]), layout=(24, 256), shape=(24, 256)
            )
            with scope[:24] as scope_1:
                with scope_1 as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.copy(tile_2, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(xf[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.cast(src_frame, dst_frame_1, dtype="f32")
                    lhs_frame = T.tensor_view(
                        T.ptr_of(xf[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(xf[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.binary(lhs_frame, rhs_frame, dst_frame_2, kind=BinaryKind.MUL)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads.d0, 2, 32 @ threads.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(ms[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    T.reduce(
                        src_frame_1,
                        dst_frame_3,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MEAN,
                    )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1e-06)
                with scope_1 as threads_4:
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(ms[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads_4.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads_4.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    T.binary(lhs_frame_1, scalar, dst_frame_4, kind=BinaryKind.ADD)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads_4.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads_4.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    T.unary(src_frame_2, dst_frame_5, kind=UnaryKind.RSQRT)
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(xf[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_4.d0, 2, 32 @ threads_4.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 24, 0:0 + 1]),
                        layout=((24 @ threads_4.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(24, 1),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_4.d0, 2, 32 @ threads_4.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.binary(lhs_frame_2, rhs_frame_1, dst_frame_6, kind=BinaryKind.MUL)
                tile_3 = T.tensor_view(
                    T.ptr_of(gamma_q[0:0 + 256]), layout=(1, 256), shape=(1, 256)
                )
                with scope_1 as threads_7:
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.copy(tile_3, dst_frame_7)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_7.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.cast(src_frame_3, dst_frame_8, dtype="f32")
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                with scope_1 as threads_9:
                    lhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.binary(lhs_frame_3, scalar_1, dst_frame_9, kind=BinaryKind.ADD)
                    lhs_frame_4 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_9.d0, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    rhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_9.d0, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.binary(lhs_frame_4, rhs_frame_2, dst_frame_10, kind=BinaryKind.MUL)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_9.d0, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(n[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_9.d0, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.cast(src_frame_4, dst_frame_11, dtype="bf16")
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(n[0:0 + 24, 0:0 + 256]),
                        layout=((24 @ threads_9.d0, 2, 32 @ threads_9.d1, 4), (0, 4, 0, 1)),
                        shape=(24, 256),
                    )
                    T.copy(src_frame_5, value_7)
                tile_4 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 24, 0:0 + 32]),
                    layout=((24, 32), (256, 1)),
                    shape=(24, 32),
                )
                with scope_1 as threads_13:
                    dst_frame_12 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_13.d0, 32 @ threads_13.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.copy(tile_4, dst_frame_12)
                    src_frame_6 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_13.d0, 32 @ threads_13.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_13 = T.tensor_view(
                        T.ptr_of(lo[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_13.d0, 32 @ threads_13.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_6, dst_frame_13, dtype="f32")
                tile_5 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 0:0 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_1 as threads_15:
                    dst_frame_14 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_15.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_5, dst_frame_14)
                    src_frame_7 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_15.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_15 = T.tensor_view(
                        T.ptr_of(c_lo[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_15.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_7, dst_frame_15, dtype="f32")
                    lhs_frame_5 = T.tensor_view(
                        T.ptr_of(lo[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_3 = T.tensor_view(
                        T.ptr_of(c_lo[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_15.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_16 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_5, rhs_frame_3, dst_frame_16, kind=BinaryKind.MUL)
                    src_frame_8 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_17 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_8, dst_frame_17, dtype="bf16")
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_18 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_15.d0, 32 @ threads_15.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_9, dst_frame_18, dtype="f32")
                tile_6 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 24, 32:32 + 32]),
                    layout=((24, 32), (256, 1)) + 32,
                    shape=(24, 32),
                )
                with scope_1 as threads_20:
                    dst_frame_19 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_20.d0, 32 @ threads_20.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.copy(tile_6, dst_frame_19)
                    src_frame_10 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_20.d0, 32 @ threads_20.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_20 = T.tensor_view(
                        T.ptr_of(hi[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_20.d0, 32 @ threads_20.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_10, dst_frame_20, dtype="f32")
                    src_frame_11 = T.tensor_view(
                        T.ptr_of(hi[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_20.d0, 32 @ threads_20.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_21 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_20.d0, 32 @ threads_20.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.unary(src_frame_11, dst_frame_21, kind=UnaryKind.NEG)
                tile_7 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 0:0 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_1 as threads_23:
                    dst_frame_22 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_23.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_7, dst_frame_22)
                    src_frame_12 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_23.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_23 = T.tensor_view(
                        T.ptr_of(s_lo[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_23.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_12, dst_frame_23, dtype="f32")
                    lhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_4 = T.tensor_view(
                        T.ptr_of(s_lo[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_23.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_24 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_6, rhs_frame_4, dst_frame_24, kind=BinaryKind.MUL)
                    src_frame_13 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_25 = T.tensor_view(
                        T.ptr_of(value_19[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_13, dst_frame_25, dtype="bf16")
                    src_frame_14 = T.tensor_view(
                        T.ptr_of(value_19[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_26 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_14, dst_frame_26, dtype="f32")
                    lhs_frame_7 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_5 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_27 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_7, rhs_frame_5, dst_frame_27, kind=BinaryKind.ADD)
                    src_frame_15 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_28 = T.tensor_view(
                        T.ptr_of(r_lo[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_23.d0, 32 @ threads_23.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_15, dst_frame_28, dtype="bf16")
                tile_8 = T.tensor_view(
                    T.ptr_of(r_lo[0:0 + 24, 0:0 + 32]),
                    layout=((1, 24, 32), (0, 0, 0)),
                    shape=(1, 24, 32),
                )
                with scope_1 as threads_30:
                    T.copy(tile_8, value_1)
                window = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 0:0 + 24, 0:0 + 32]),
                    layout=((1, 24, 32), (8192, 256, 1)),
                    shape=(1, 24, 32),
                )
                T.copy(value_1, window)
                tile_9 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 32:32 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_1 as threads_31:
                    dst_frame_29 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_31.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_9, dst_frame_29)
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_31.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_30 = T.tensor_view(
                        T.ptr_of(c_hi[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_31.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_16, dst_frame_30, dtype="f32")
                    lhs_frame_8 = T.tensor_view(
                        T.ptr_of(hi[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_6 = T.tensor_view(
                        T.ptr_of(c_hi[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_31.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_31 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_8, rhs_frame_6, dst_frame_31, kind=BinaryKind.MUL)
                    src_frame_17 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_32 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_17, dst_frame_32, dtype="bf16")
                    src_frame_18 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_33 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_31.d0, 32 @ threads_31.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_18, dst_frame_33, dtype="f32")
                tile_10 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 32:32 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_1 as threads_36:
                    dst_frame_34 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_36.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_10, dst_frame_34)
                    src_frame_19 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_36.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_35 = T.tensor_view(
                        T.ptr_of(s_hi[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_36.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_19, dst_frame_35, dtype="f32")
                    lhs_frame_9 = T.tensor_view(
                        T.ptr_of(lo[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_7 = T.tensor_view(
                        T.ptr_of(s_hi[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_36.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_36 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_9, rhs_frame_7, dst_frame_36, kind=BinaryKind.MUL)
                    src_frame_20 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_37 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_20, dst_frame_37, dtype="bf16")
                    src_frame_21 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_38 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_21, dst_frame_38, dtype="f32")
                    lhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    rhs_frame_8 = T.tensor_view(
                        T.ptr_of(value_30[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_39 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.binary(lhs_frame_10, rhs_frame_8, dst_frame_39, kind=BinaryKind.ADD)
                    src_frame_22 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    dst_frame_40 = T.tensor_view(
                        T.ptr_of(r_hi[0:0 + 24, 0:0 + 32]),
                        layout=((24 @ threads_36.d0, 32 @ threads_36.d1), (0, 0)),
                        shape=(24, 32),
                    )
                    T.cast(src_frame_22, dst_frame_40, dtype="bf16")
                tile_11 = T.tensor_view(
                    T.ptr_of(r_hi[0:0 + 24, 0:0 + 32]),
                    layout=((1, 24, 32), (0, 0, 0)),
                    shape=(1, 24, 32),
                )
                with scope_1 as threads_43:
                    T.copy(tile_11, value_24_1)
                window_1 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 0:0 + 24, 32:32 + 32]),
                    layout=((1, 24, 32), (8192, 256, 1)),
                    shape=(1, 24, 32),
                )
                T.copy(value_24_1, window_1)
                tile_12 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 24, 64:64 + 192]),
                    layout=((24, 192), (256, 1)) + 64,
                    shape=(24, 192),
                )
                tile_13 = T.tensor_view(
                    T.ptr_of(tile_12[0:0 + 24, 0:0 + 192]),
                    layout=(1, 24, 192) + 64,
                    shape=(1, 24, 192),
                )
                with scope_1 as threads_44:
                    T.copy(tile_13, value_34_1)
                window_2 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 0:0 + 24, 64:64 + 192]),
                    layout=((1, 24, 192), (8192, 256, 1)),
                    shape=(1, 24, 192),
                )
                T.copy(value_34_1, window_2)
            tile_14 = T.tensor_view(
                T.ptr_of(qkv[cta.d0:cta.d0 + 1, 12288:12288 + 1024]),
                layout=((1, 1024), (14336, 1)),
                shape=(1, 1024),
            )
            tile_15 = T.tensor_view(
                T.ptr_of(tile_14[0:0 + 1, 0:0 + 1024]), layout=(4, 256), shape=(4, 256)
            )
            with scope[24:28] as scope_2:
                with scope_2 as threads_45:
                    dst_frame_41 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.copy(tile_15, dst_frame_41)
                    src_frame_23 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    dst_frame_42 = T.tensor_view(
                        T.ptr_of(xf_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.cast(src_frame_23, dst_frame_42, dtype="f32")
                    lhs_frame_11 = T.tensor_view(
                        T.ptr_of(xf_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    rhs_frame_9 = T.tensor_view(
                        T.ptr_of(xf_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    dst_frame_43 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.binary(lhs_frame_11, rhs_frame_9, dst_frame_43, kind=BinaryKind.MUL)
                    src_frame_24 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_45.d0, 2, 32 @ threads_45.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    dst_frame_44 = T.tensor_view(
                        T.ptr_of(ms_1[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_45.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    T.reduce(
                        src_frame_24,
                        dst_frame_44,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.MEAN,
                    )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1e-06)
                with scope_2 as threads_49:
                    lhs_frame_12 = T.tensor_view(
                        T.ptr_of(ms_1[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_49.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    dst_frame_45 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_49.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    T.binary(lhs_frame_12, scalar_2, dst_frame_45, kind=BinaryKind.ADD)
                    src_frame_25 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_49.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    dst_frame_46 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_49.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    T.unary(src_frame_25, dst_frame_46, kind=UnaryKind.RSQRT)
                    lhs_frame_13 = T.tensor_view(
                        T.ptr_of(xf_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_49.d0, 2, 32 @ threads_49.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    rhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 4, 0:0 + 1]),
                        layout=((4 @ threads_49.d0, 1, 1, 1), (1, 0, 0, 0)),
                        shape=(4, 1),
                    )
                    dst_frame_47 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_49.d0, 2, 32 @ threads_49.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.binary(lhs_frame_13, rhs_frame_10, dst_frame_47, kind=BinaryKind.MUL)
                tile_16 = T.tensor_view(
                    T.ptr_of(gamma_k[0:0 + 256]), layout=(1, 256), shape=(1, 256)
                )
                with scope_2 as threads_52:
                    dst_frame_48 = T.tensor_view(
                        T.ptr_of(value_50[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.copy(tile_16, dst_frame_48)
                    src_frame_26 = T.tensor_view(
                        T.ptr_of(value_50[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_49 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.cast(src_frame_26, dst_frame_49, dtype="f32")
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 1.0)
                with scope_2 as threads_54:
                    lhs_frame_14 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_50 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    T.binary(lhs_frame_14, scalar_3, dst_frame_50, kind=BinaryKind.ADD)
                    lhs_frame_15 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_54.d0, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    rhs_frame_11 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 1, 0:0 + 256]),
                        layout=((1, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(1, 256),
                    )
                    dst_frame_51 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_54.d0, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.binary(lhs_frame_15, rhs_frame_11, dst_frame_51, kind=BinaryKind.MUL)
                    src_frame_27 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_54.d0, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    dst_frame_52 = T.tensor_view(
                        T.ptr_of(n_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_54.d0, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.cast(src_frame_27, dst_frame_52, dtype="bf16")
                    src_frame_28 = T.tensor_view(
                        T.ptr_of(n_1[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_54.d0, 2, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(4, 256),
                    )
                    T.copy(src_frame_28, value_41)
                tile_17 = T.tensor_view(
                    T.ptr_of(value_41[0:0 + 4, 0:0 + 32]),
                    layout=((4, 32), (256, 1)),
                    shape=(4, 32),
                )
                with scope_2 as threads_58:
                    dst_frame_53 = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_58.d0, 32 @ threads_58.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.copy(tile_17, dst_frame_53)
                    src_frame_29 = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_58.d0, 32 @ threads_58.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_54 = T.tensor_view(
                        T.ptr_of(lo_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_58.d0, 32 @ threads_58.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_29, dst_frame_54, dtype="f32")
                tile_18 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 0:0 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_2 as threads_60:
                    dst_frame_55 = T.tensor_view(
                        T.ptr_of(value_51[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_60.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_18, dst_frame_55)
                    src_frame_30 = T.tensor_view(
                        T.ptr_of(value_51[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_60.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_56 = T.tensor_view(
                        T.ptr_of(c_lo_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_60.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_30, dst_frame_56, dtype="f32")
                    lhs_frame_16 = T.tensor_view(
                        T.ptr_of(lo_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_12 = T.tensor_view(
                        T.ptr_of(c_lo_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_60.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_57 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_16, rhs_frame_12, dst_frame_57, kind=BinaryKind.MUL)
                    src_frame_31 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_58 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_31, dst_frame_58, dtype="bf16")
                    src_frame_32 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_59 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_60.d0, 32 @ threads_60.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_32, dst_frame_59, dtype="f32")
                tile_19 = T.tensor_view(
                    T.ptr_of(value_41[0:0 + 4, 32:32 + 32]),
                    layout=((4, 32), (256, 1)) + 32,
                    shape=(4, 32),
                )
                with scope_2 as threads_65:
                    dst_frame_60 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_65.d0, 32 @ threads_65.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.copy(tile_19, dst_frame_60)
                    src_frame_33 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_65.d0, 32 @ threads_65.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_61 = T.tensor_view(
                        T.ptr_of(hi_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_65.d0, 32 @ threads_65.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_33, dst_frame_61, dtype="f32")
                    src_frame_34 = T.tensor_view(
                        T.ptr_of(hi_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_65.d0, 32 @ threads_65.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_62 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_65.d0, 32 @ threads_65.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.unary(src_frame_34, dst_frame_62, kind=UnaryKind.NEG)
                tile_20 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 0:0 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_2 as threads_68:
                    dst_frame_63 = T.tensor_view(
                        T.ptr_of(value_57[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_68.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_20, dst_frame_63)
                    src_frame_35 = T.tensor_view(
                        T.ptr_of(value_57[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_68.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_64 = T.tensor_view(
                        T.ptr_of(s_lo_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_68.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_35, dst_frame_64, dtype="f32")
                    lhs_frame_17 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_13 = T.tensor_view(
                        T.ptr_of(s_lo_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_68.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_65 = T.tensor_view(
                        T.ptr_of(value_54[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_17, rhs_frame_13, dst_frame_65, kind=BinaryKind.MUL)
                    src_frame_36 = T.tensor_view(
                        T.ptr_of(value_54[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_66 = T.tensor_view(
                        T.ptr_of(value_53[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_36, dst_frame_66, dtype="bf16")
                    src_frame_37 = T.tensor_view(
                        T.ptr_of(value_53[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_67 = T.tensor_view(
                        T.ptr_of(value_52[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_37, dst_frame_67, dtype="f32")
                    lhs_frame_18 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_14 = T.tensor_view(
                        T.ptr_of(value_52[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_68 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_18, rhs_frame_14, dst_frame_68, kind=BinaryKind.ADD)
                    src_frame_38 = T.tensor_view(
                        T.ptr_of(value_36[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_69 = T.tensor_view(
                        T.ptr_of(r_lo_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_68.d0, 32 @ threads_68.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_38, dst_frame_69, dtype="bf16")
                tile_21 = T.tensor_view(
                    T.ptr_of(r_lo_1[0:0 + 4, 0:0 + 32]),
                    layout=((1, 4, 32), (0, 0, 0)),
                    shape=(1, 4, 32),
                )
                with scope_2 as threads_75:
                    T.copy(tile_21, value_35_1)
                window_3 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 24:24 + 4, 0:0 + 32]),
                    layout=((1, 4, 32), (8192, 256, 1)),
                    shape=(1, 4, 32),
                )
                T.copy(value_35_1, window_3)
                tile_22 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 32:32 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_2 as threads_76:
                    dst_frame_70 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_76.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_22, dst_frame_70)
                    src_frame_39 = T.tensor_view(
                        T.ptr_of(value_63[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_76.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_71 = T.tensor_view(
                        T.ptr_of(c_hi_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_76.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_39, dst_frame_71, dtype="f32")
                    lhs_frame_19 = T.tensor_view(
                        T.ptr_of(hi_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_15 = T.tensor_view(
                        T.ptr_of(c_hi_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_76.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_72 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_19, rhs_frame_15, dst_frame_72, kind=BinaryKind.MUL)
                    src_frame_40 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_73 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_40, dst_frame_73, dtype="bf16")
                    src_frame_41 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_74 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_76.d0, 32 @ threads_76.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_41, dst_frame_74, dtype="f32")
                tile_23 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 32:32 + 32]),
                    layout=((1, 32), (64, 1)),
                    shape=(1, 32),
                )
                with scope_2 as threads_81:
                    dst_frame_75 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_81.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.copy(tile_23, dst_frame_75)
                    src_frame_42 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_81.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_76 = T.tensor_view(
                        T.ptr_of(s_hi_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_81.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    T.cast(src_frame_42, dst_frame_76, dtype="f32")
                    lhs_frame_20 = T.tensor_view(
                        T.ptr_of(lo_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_16 = T.tensor_view(
                        T.ptr_of(s_hi_1[0:0 + 1, 0:0 + 32]),
                        layout=((1, 32 @ threads_81.d1), (0, 0)),
                        shape=(1, 32),
                    )
                    dst_frame_77 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_20, rhs_frame_16, dst_frame_77, kind=BinaryKind.MUL)
                    src_frame_43 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_78 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_43, dst_frame_78, dtype="bf16")
                    src_frame_44 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_79 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_44, dst_frame_79, dtype="f32")
                    lhs_frame_21 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    rhs_frame_17 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_80 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.binary(lhs_frame_21, rhs_frame_17, dst_frame_80, kind=BinaryKind.ADD)
                    src_frame_45 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    dst_frame_81 = T.tensor_view(
                        T.ptr_of(r_hi_1[0:0 + 4, 0:0 + 32]),
                        layout=((4 @ threads_81.d0, 32 @ threads_81.d1), (0, 0)),
                        shape=(4, 32),
                    )
                    T.cast(src_frame_45, dst_frame_81, dtype="bf16")
                tile_24 = T.tensor_view(
                    T.ptr_of(r_hi_1[0:0 + 4, 0:0 + 32]),
                    layout=((1, 4, 32), (0, 0, 0)),
                    shape=(1, 4, 32),
                )
                with scope_2 as threads_88:
                    T.copy(tile_24, value_58_1)
                window_4 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 24:24 + 4, 32:32 + 32]),
                    layout=((1, 4, 32), (8192, 256, 1)),
                    shape=(1, 4, 32),
                )
                T.copy(value_58_1, window_4)
                tile_25 = T.tensor_view(
                    T.ptr_of(value_41[0:0 + 4, 64:64 + 192]),
                    layout=((4, 192), (256, 1)) + 64,
                    shape=(4, 192),
                )
                tile_26 = T.tensor_view(
                    T.ptr_of(tile_25[0:0 + 4, 0:0 + 192]),
                    layout=(1, 4, 192) + 64,
                    shape=(1, 4, 192),
                )
                with scope_2 as threads_89:
                    T.copy(tile_26, value_68_1)
                window_5 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 24:24 + 4, 64:64 + 192]),
                    layout=((1, 4, 192), (8192, 256, 1)),
                    shape=(1, 4, 192),
                )
                T.copy(value_68_1, window_5)
            with scope[28:] as scope_3:
                tile_27 = T.tensor_view(
                    T.ptr_of(qkv[cta.d0:cta.d0 + 1, 13312:13312 + 1024]),
                    layout=((1, 1024), (14336, 1)),
                    shape=(1, 1024),
                )
                tile_28 = T.tensor_view(
                    T.ptr_of(tile_27[0:0 + 1, 0:0 + 1024]), layout=(4, 256), shape=(4, 256)
                )
                with scope_3 as threads_90:
                    dst_frame_82 = T.tensor_view(
                        T.ptr_of(value_70[0:0 + 4, 0:0 + 256]),
                        layout=((4 @ threads_90.d0, 32 @ threads_90.d1, 8), (0, 0, 1)),
                        shape=(4, 256),
                    )
                    T.copy(tile_28, dst_frame_82)
                tile_29 = T.tensor_view(
                    T.ptr_of(value_70[0:0 + 4, 0:0 + 256]),
                    layout=((1, 4, 32, 8), (0, 0, 0, 1)),
                    shape=(1, 4, 256),
                )
                with scope_3 as threads_91:
                    T.copy(tile_29, value_69_1)
                window_6 = T.tensor_view(
                    T.ptr_of(out[cta.d0:cta.d0 + 1, 28:28 + 4, 0:0 + 256]),
                    layout=((1, 4, 256), (8192, 256, 1)),
                    shape=(1, 4, 256),
                )
                T.copy(value_69_1, window_6)
