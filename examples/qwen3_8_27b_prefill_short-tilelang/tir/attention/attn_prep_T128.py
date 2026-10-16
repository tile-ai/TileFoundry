# analysis target=nvidia.h200_sxm module=SCHED_ATTN_PREP_T128 function=attn_prep topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r33.50KB/w2.03MB@logical,r4.19MB/w260.00MB@total,r33.50KB/w2.03MB@cta,r16.09KB/w2.03MB@thread;rmem:r271.79KB/w208.33KB@logical,r33.97MB/w26.04MB@total,r271.79KB/w208.33KB@cta,r1.29KB/w704B@thread;smem:r14.00KB/w14.00KB@logical,r1.75MB/w1.75MB@total,r14.00KB/w14.00KB@cta,r10.51KB/w32B@thread footprint=cos:16.00KB;gamma_k:512B;gamma_q:512B;qkv:2.00MB;sin:16.00KB;v122:79:256B;v124:82:32.00KB;v142:80:256B;v144:83:32.00KB;v147:81:1.50KB;v149:84:192.00KB;v156:87:2.00KB;v158:88:256.00KB;v48:57:1.50KB;v50:60:192.00KB;v68:58:1.50KB;v70:61:192.00KB;v73:59:9.00KB;v75:62:1.12MB footprint-precision=exact peak=gmem:5.54MB;rmem:62.60KB;smem:12.00KB persistent=gmem:3.53MB
#   buffer=gamma_q holds=4.05MB time=none space=cta.t reuse=63.50KB fits=yes precision=exact
#   buffer=gamma_k holds=4.05MB time=none space=cta.t reuse=63.50KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_prep(
    qkv: Tensor[(128, 14336), "bf16"], gamma_q: Tensor[(256,), "bf16"], gamma_k: Tensor[(256,), "bf16"], cos: Tensor[(128, 64), "bf16"], sin: Tensor[(128, 64), "bf16"], scratch: Tensor[(768,), "bf16"], scratch_1: Tensor[(768,), "bf16"], scratch_2: Tensor[(4608,), "bf16"], scratch_3: Tensor[(128,), "bf16"], scratch_4: Tensor[(128,), "bf16"], scratch_5: Tensor[(768,), "bf16"], scratch_6: Tensor[(1024,), "bf16"], out: Tensor[(128, 32, 256), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(128,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1024),), layout=(32, 32), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_3 = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((24, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xf = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((24, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            ms = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((24, 1, 1, 1), (1, 0, 0, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((1, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            n = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((24, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            lo = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((24, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            c_lo = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            hi = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((24, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            s_lo = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            r_lo = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 32),
                    "bf16",
                    ShardLayout(
                        layout=Layout((24, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            c_hi = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            s_hi = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            r_hi = T.alloc_tensor(
                tensor_type=Tensor[
                    (24, 32),
                    "bf16",
                    ShardLayout(
                        layout=Layout((24, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 1024),), layout=(24, 32), names=("d0", "d1")
            ) as scope_1:
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                value_4 = T.tensor_view(
                    T.ptr_of(qkv[cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((24, 256), (512, 1)),
                    shape=(24, 256),
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_3),
                    layout=((24 @ scope_1.d0, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)),
                    shape=(24, 256),
                )
                T.copy(value_4, value_5)
                T.cast(value_5, xf, dtype="f32")
                value_6 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 256),
                        "f32",
                        ((24 @ scope_1.d0, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xf, xf, value_6, kind=BinaryKind.MUL)
                T.reduce(value_6, ms, axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
                value_7 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 1), "f32", ((24 @ scope_1.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1e-06)
                T.binary(ms, scalar, value_7, kind=BinaryKind.ADD)
                value_8 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 1), "f32", ((24 @ scope_1.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                    ]
                )
                T.unary(value_7, value_8, kind=UnaryKind.RSQRT)
                value_9 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 256),
                        "f32",
                        ((24 @ scope_1.d0, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xf, value_8, value_9, kind=BinaryKind.MUL)
                value_11 = T.tensor_view(
                    T.ptr_of(gamma_q), layout=((1, 256), (0, 1)), shape=(1, 256)
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_10),
                    layout=((1, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 256),
                )
                T.copy(value_11, value_12)
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 256), "f32", ((1, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)), "rmem"
                    ]
                )
                T.cast(value_12, value_13, dtype="f32")
                value_14 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 256), "f32", ((1, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(value_13, scalar_1, value_14, kind=BinaryKind.ADD)
                value_15 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 256),
                        "f32",
                        ((24 @ scope_1.d0, 2, 32 @ scope_1.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_9, value_14, value_15, kind=BinaryKind.MUL)
                T.cast(value_15, n, dtype="bf16")
                value_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(24, 256), shape=(24, 256)
                )
                value_16 = T.tensor_view(T.ptr_of(value_2), layout=(24, 256), shape=(24, 256))
                T.copy(n, value_16)
                value_17 = T.tensor_view(
                    T.ptr_of(value_2), layout=((24, 32), (256, 1)), shape=(24, 32)
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)),
                    shape=(24, 32),
                )
                T.copy(value_17, value_18)
                T.cast(value_18, lo, dtype="f32")
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_1.d1), (0, 0)), "rmem"]
                )
                value_20 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_19), layout=((1, 32 @ scope_1.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_20, value_21)
                T.cast(value_21, c_lo, dtype="f32")
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(lo, c_lo, value_22, kind=BinaryKind.MUL)
                value_23 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_22, value_23, dtype="bf16")
                value_24 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_23, value_24, dtype="f32")
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                value_26 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 32:32 + 1]),
                    layout=((24, 32), (256, 1)),
                    shape=(24, 32),
                )
                value_27 = T.tensor_view(
                    T.ptr_of(value_25),
                    layout=((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)),
                    shape=(24, 32),
                )
                T.copy(value_26, value_27)
                T.cast(value_27, hi, dtype="f32")
                value_28 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.unary(hi, value_28, kind=UnaryKind.NEG)
                value_29 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_1.d1), (0, 0)), "rmem"]
                )
                value_30 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_31 = T.tensor_view(
                    T.ptr_of(value_29), layout=((1, 32 @ scope_1.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_30, value_31)
                T.cast(value_31, s_lo, dtype="f32")
                value_32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_28, s_lo, value_32, kind=BinaryKind.MUL)
                value_33 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_32, value_33, dtype="bf16")
                value_34 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_33, value_34, dtype="f32")
                value_35 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_24, value_34, value_35, kind=BinaryKind.ADD)
                T.cast(value_35, r_lo, dtype="bf16")
                value_36 = T.tensor_view(
                    T.ptr_of(r_lo),
                    layout=((1, 24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0, 0)),
                    shape=(1, 24, 32),
                )
                value_37 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 768]), layout=(1, 24, 32), shape=(1, 24, 32)
                )
                value_38 = T.tensor_view(
                    T.ptr_of(value_37), layout=(1, 24, 32), shape=(1, 24, 32)
                )
                T.copy(value_36, value_38)
                value_39 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window = T.tensor_view(
                    T.ptr_of(value_39[cta.d0:cta.d0 + 1, 0:0 + 24, 0:0 + 32]),
                    layout=((1, 24, 32), (8192, 256, 1)),
                    shape=(1, 24, 32),
                )
                T.copy(value_38, window)
                value_41 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_1.d1), (0, 0)), "rmem"]
                )
                value_42 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 32:32 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_43 = T.tensor_view(
                    T.ptr_of(value_41), layout=((1, 32 @ scope_1.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_42, value_43)
                T.cast(value_43, c_hi, dtype="f32")
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(hi, c_hi, value_44, kind=BinaryKind.MUL)
                value_45 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_44, value_45, dtype="bf16")
                value_46 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_45, value_46, dtype="f32")
                value_47 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_1.d1), (0, 0)), "rmem"]
                )
                value_48 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 32:32 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_49 = T.tensor_view(
                    T.ptr_of(value_47), layout=((1, 32 @ scope_1.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_48, value_49)
                T.cast(value_49, s_hi, dtype="f32")
                value_50 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(lo, s_hi, value_50, kind=BinaryKind.MUL)
                value_51 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "bf16", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_50, value_51, dtype="bf16")
                value_52 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_51, value_52, dtype="f32")
                value_53 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (24, 32), "f32", ((24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_46, value_52, value_53, kind=BinaryKind.ADD)
                T.cast(value_53, r_hi, dtype="bf16")
                value_54 = T.tensor_view(
                    T.ptr_of(r_hi),
                    layout=((1, 24 @ scope_1.d0, 32 @ scope_1.d1), (0, 0, 0)),
                    shape=(1, 24, 32),
                )
                value_40_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 768]), layout=(1, 24, 32), shape=(1, 24, 32)
                )
                value_55 = T.tensor_view(
                    T.ptr_of(value_40_1), layout=(1, 24, 32), shape=(1, 24, 32)
                )
                T.copy(value_54, value_55)
                value_56 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_56[cta.d0:cta.d0 + 1, 0:0 + 24, 32:32 + 32]),
                    layout=((1, 24, 32), (8192, 256, 1)),
                    shape=(1, 24, 32),
                )
                T.copy(value_55, window_1)
                value_58 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 1, 64:64 + 1]),
                    layout=((1, 24, 192), (0, 256, 1)),
                    shape=(1, 24, 192),
                )
                value_57_1 = T.tensor_view(
                    T.ptr_of(scratch_2[0:0 + 4608]), layout=(1, 24, 192), shape=(1, 24, 192)
                )
                value_59 = T.tensor_view(
                    T.ptr_of(value_57_1), layout=(1, 24, 192), shape=(1, 24, 192)
                )
                T.copy(value_58, value_59)
                value_60 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_60[cta.d0:cta.d0 + 1, 0:0 + 24, 64:64 + 192]),
                    layout=((1, 24, 192), (8192, 256, 1)),
                    shape=(1, 24, 192),
                )
                T.copy(value_59, window_2)
            value_64 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            xf_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            ms_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 1, 1, 1), (1, 0, 0, 0)),
                        attrs=(S(0), B()),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            value_71 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((1, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(B(), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            n_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 256),
                    "bf16",
                    ShardLayout(
                        layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                        attrs=(S(0), S(2)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            lo_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            c_lo_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            hi_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            s_lo_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            r_lo_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 32),
                    "bf16",
                    ShardLayout(
                        layout=Layout((4, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            c_hi_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            s_hi_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((1, 32), (0, 0)),
                        attrs=(B(), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            r_hi_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 32),
                    "bf16",
                    ShardLayout(
                        layout=Layout((4, 32), (0, 0)),
                        attrs=(S(0), S(1)),
                        mesh=Mesh((Topology("thread", 1024),), layout=ComposedLayout(
                            inner=None,
                            offset=768,
                            outer=Layout((4, 32), (32, 1)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with scope[24:28] as scope_2:
                value_62 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                value_65 = T.tensor_view(
                    T.ptr_of(qkv[cta.d0:cta.d0 + 1, 12288:12288 + 1]),
                    layout=(4, 256),
                    shape=(4, 256),
                )
                value_66 = T.tensor_view(
                    T.ptr_of(value_64),
                    layout=((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(4, 256),
                )
                T.copy(value_65, value_66)
                T.cast(value_66, xf_1, dtype="f32")
                value_67 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 256),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xf_1, xf_1, value_67, kind=BinaryKind.MUL)
                T.reduce(value_67, ms_1, axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
                value_68 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 1), "f32", ((4 @ scope_2.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1e-06)
                T.binary(ms_1, scalar_2, value_68, kind=BinaryKind.ADD)
                value_69 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 1), "f32", ((4 @ scope_2.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                    ]
                )
                T.unary(value_68, value_69, kind=UnaryKind.RSQRT)
                value_70 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 256),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xf_1, value_69, value_70, kind=BinaryKind.MUL)
                value_72 = T.tensor_view(
                    T.ptr_of(gamma_k), layout=((1, 256), (0, 1)), shape=(1, 256)
                )
                value_73 = T.tensor_view(
                    T.ptr_of(value_71),
                    layout=((1, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(1, 256),
                )
                T.copy(value_72, value_73)
                value_74 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 256), "f32", ((1, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)), "rmem"
                    ]
                )
                T.cast(value_73, value_74, dtype="f32")
                value_75 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (1, 256), "f32", ((1, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)), "rmem"
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 1.0)
                T.binary(value_74, scalar_3, value_75, kind=BinaryKind.ADD)
                value_76 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 256),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_70, value_75, value_76, kind=BinaryKind.MUL)
                T.cast(value_76, n_1, dtype="bf16")
                value_63 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(4, 256), shape=(4, 256)
                )
                value_77 = T.tensor_view(T.ptr_of(value_63), layout=(4, 256), shape=(4, 256))
                T.copy(n_1, value_77)
                value_78 = T.tensor_view(
                    T.ptr_of(value_63), layout=((4, 32), (256, 1)), shape=(4, 32)
                )
                value_79 = T.tensor_view(
                    T.ptr_of(value_62),
                    layout=((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)),
                    shape=(4, 32),
                )
                T.copy(value_78, value_79)
                T.cast(value_79, lo_1, dtype="f32")
                value_80 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_2.d1), (0, 0)), "rmem"]
                )
                value_81 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_82 = T.tensor_view(
                    T.ptr_of(value_80), layout=((1, 32 @ scope_2.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_81, value_82)
                T.cast(value_82, c_lo_1, dtype="f32")
                value_83 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(lo_1, c_lo_1, value_83, kind=BinaryKind.MUL)
                value_84 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_83, value_84, dtype="bf16")
                value_85 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_84, value_85, dtype="f32")
                value_86 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                value_87 = T.tensor_view(
                    T.ptr_of(value_63[0:0 + 1, 32:32 + 1]),
                    layout=((4, 32), (256, 1)),
                    shape=(4, 32),
                )
                value_88 = T.tensor_view(
                    T.ptr_of(value_86),
                    layout=((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)),
                    shape=(4, 32),
                )
                T.copy(value_87, value_88)
                T.cast(value_88, hi_1, dtype="f32")
                value_89 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.unary(hi_1, value_89, kind=UnaryKind.NEG)
                value_90 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_2.d1), (0, 0)), "rmem"]
                )
                value_91 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 0:0 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_92 = T.tensor_view(
                    T.ptr_of(value_90), layout=((1, 32 @ scope_2.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_91, value_92)
                T.cast(value_92, s_lo_1, dtype="f32")
                value_93 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_89, s_lo_1, value_93, kind=BinaryKind.MUL)
                value_94 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_93, value_94, dtype="bf16")
                value_95 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_94, value_95, dtype="f32")
                value_96 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_85, value_95, value_96, kind=BinaryKind.ADD)
                T.cast(value_96, r_lo_1, dtype="bf16")
                value_97 = T.tensor_view(
                    T.ptr_of(r_lo_1),
                    layout=((1, 4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0, 0)),
                    shape=(1, 4, 32),
                )
                value_61_1 = T.tensor_view(
                    T.ptr_of(scratch_3[0:0 + 128]), layout=(1, 4, 32), shape=(1, 4, 32)
                )
                value_98 = T.tensor_view(
                    T.ptr_of(value_61_1), layout=(1, 4, 32), shape=(1, 4, 32)
                )
                T.copy(value_97, value_98)
                value_99 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_3 = T.tensor_view(
                    T.ptr_of(value_99[cta.d0:cta.d0 + 1, 24:24 + 4, 0:0 + 32]),
                    layout=((1, 4, 32), (8192, 256, 1)),
                    shape=(1, 4, 32),
                )
                T.copy(value_98, window_3)
                value_101 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_2.d1), (0, 0)), "rmem"]
                )
                value_102 = T.tensor_view(
                    T.ptr_of(cos[cta.d0:cta.d0 + 1, 32:32 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_103 = T.tensor_view(
                    T.ptr_of(value_101), layout=((1, 32 @ scope_2.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_102, value_103)
                T.cast(value_103, c_hi_1, dtype="f32")
                value_104 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(hi_1, c_hi_1, value_104, kind=BinaryKind.MUL)
                value_105 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_104, value_105, dtype="bf16")
                value_106 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_105, value_106, dtype="f32")
                value_107 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 32), "bf16", ((1, 32 @ scope_2.d1), (0, 0)), "rmem"]
                )
                value_108 = T.tensor_view(
                    T.ptr_of(sin[cta.d0:cta.d0 + 1, 32:32 + 1]),
                    layout=((1, 32), (0, 1)),
                    shape=(1, 32),
                )
                value_109 = T.tensor_view(
                    T.ptr_of(value_107), layout=((1, 32 @ scope_2.d1), (0, 0)), shape=(1, 32)
                )
                T.copy(value_108, value_109)
                T.cast(value_109, s_hi_1, dtype="f32")
                value_110 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(lo_1, s_hi_1, value_110, kind=BinaryKind.MUL)
                value_111 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "bf16", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_110, value_111, dtype="bf16")
                value_112 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.cast(value_111, value_112, dtype="f32")
                value_113 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", ((4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0)), "rmem"
                    ]
                )
                T.binary(value_106, value_112, value_113, kind=BinaryKind.ADD)
                T.cast(value_113, r_hi_1, dtype="bf16")
                value_114 = T.tensor_view(
                    T.ptr_of(r_hi_1),
                    layout=((1, 4 @ scope_2.d0, 32 @ scope_2.d1), (0, 0, 0)),
                    shape=(1, 4, 32),
                )
                value_100_1 = T.tensor_view(
                    T.ptr_of(scratch_4[0:0 + 128]), layout=(1, 4, 32), shape=(1, 4, 32)
                )
                value_115 = T.tensor_view(
                    T.ptr_of(value_100_1), layout=(1, 4, 32), shape=(1, 4, 32)
                )
                T.copy(value_114, value_115)
                value_116 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_4 = T.tensor_view(
                    T.ptr_of(value_116[cta.d0:cta.d0 + 1, 24:24 + 4, 32:32 + 32]),
                    layout=((1, 4, 32), (8192, 256, 1)),
                    shape=(1, 4, 32),
                )
                T.copy(value_115, window_4)
                value_118 = T.tensor_view(
                    T.ptr_of(value_63[0:0 + 1, 64:64 + 1]),
                    layout=((1, 4, 192), (0, 256, 1)),
                    shape=(1, 4, 192),
                )
                value_117_1 = T.tensor_view(
                    T.ptr_of(scratch_5[0:0 + 768]), layout=(1, 4, 192), shape=(1, 4, 192)
                )
                value_119 = T.tensor_view(
                    T.ptr_of(value_117_1), layout=(1, 4, 192), shape=(1, 4, 192)
                )
                T.copy(value_118, value_119)
                value_120 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_5 = T.tensor_view(
                    T.ptr_of(value_120[cta.d0:cta.d0 + 1, 24:24 + 4, 64:64 + 192]),
                    layout=((1, 4, 192), (8192, 256, 1)),
                    shape=(1, 4, 192),
                )
                T.copy(value_119, window_5)
            with scope[28:] as scope_3:
                value_122 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 256),
                        "bf16",
                        ((4 @ scope_3.d0, 32 @ scope_3.d1, 8), (0, 0, 1)),
                        "rmem",
                    ]
                )
                value_123 = T.tensor_view(
                    T.ptr_of(qkv[cta.d0:cta.d0 + 1, 13312:13312 + 1]),
                    layout=(4, 256),
                    shape=(4, 256),
                )
                value_124 = T.tensor_view(
                    T.ptr_of(value_122),
                    layout=((4 @ scope_3.d0, 32 @ scope_3.d1, 8), (0, 0, 1)),
                    shape=(4, 256),
                )
                T.copy(value_123, value_124)
                value_125 = T.tensor_view(
                    T.ptr_of(value_122),
                    layout=((1, 4 @ scope_3.d0, 32 @ scope_3.d1, 8), (0, 0, 0, 1)),
                    shape=(1, 4, 256),
                )
                value_121_1 = T.tensor_view(
                    T.ptr_of(scratch_6[0:0 + 1024]), layout=(1, 4, 256), shape=(1, 4, 256)
                )
                value_126 = T.tensor_view(
                    T.ptr_of(value_121_1), layout=(1, 4, 256), shape=(1, 4, 256)
                )
                T.copy(value_125, value_126)
                value_127 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 32, 256), shape=(128, 32, 256)
                )
                window_6 = T.tensor_view(
                    T.ptr_of(value_127[cta.d0:cta.d0 + 1, 28:28 + 4, 0:0 + 256]),
                    layout=((1, 4, 256), (8192, 256, 1)),
                    shape=(1, 4, 256),
                )
                T.copy(value_126, window_6)
