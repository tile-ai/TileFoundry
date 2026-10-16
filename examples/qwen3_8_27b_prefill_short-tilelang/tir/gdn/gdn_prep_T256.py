# analysis target=nvidia.h200_sxm module=SCHED_GDN_PREP_T256 function=gdn_prep topology=cta wave=132/2560
# selection requested=memory executed=memory
# memory traffic=gmem:r18.02KB/w10.23MB@logical,r19.91MB/w25.55GB@total,r18.02KB/w10.23MB@cta,r15.31KB/w10.22MB@thread;rmem:r205.67KB/w189.04KB@logical,r247.51MB/w228.71MB@total,r205.67KB/w189.04KB@cta,r2.53KB/w1.83KB@thread;smem:r16.00KB/w5.50KB@logical,r20.00MB/w6.88MB@total,r16.00KB/w5.50KB@cta,r128B/w5.50KB@thread footprint=a_log:96B;conv_w:80.00KB;dt_bias:96B;proj:5.00MB;rowidx:176B;v131:133:4.00KB;v134:134:272.00KB;v159:147:32B;v162:148:2.12KB;v174:149:32B;v177:150:2.12KB;v196:154:1.50KB;v199:155:0;v65:114:4.00KB;v68:115:256.00KB footprint-precision=exact peak=gmem:18.42MB;rmem:28.05KB;smem:2.75KB persistent=gmem:8.21MB
#   buffer=proj holds=5.61MB time=none space=cta.rb reuse=5.00MB fits=yes precision=exact
#   buffer=conv_w holds=5.61MB time=none space=cta.rb reuse=80.00KB fits=yes precision=exact
#   buffer=rowidx holds=5.61MB time=none space=cta.hd reuse=8.08KB fits=yes precision=exact
#   buffer=a_log holds=5.61MB time=none space=cta.rb reuse=96B fits=yes precision=exact
#   buffer=dt_bias holds=5.61MB time=none space=cta.rb reuse=96B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_prep(
    proj: Tensor[(256, 16640), "bf16"], conv_w: Tensor[(10240, 4), "bf16", (10240, 4)], a_log: Tensor[(48,), "bf16"], dt_bias: Tensor[(48,), "bf16"], rowidx: Tensor[(352,), "i64"], scratch: Tensor[(1024,), "f32"], scratch_1: Tensor[(1024,), "f32"], scratch_2: Tensor[(8,), "f32"], scratch_3: Tensor[(8,), "f32"], scratch_4: Tensor[(384,), "f32"], out: Tensor[(259, 10336), "f32"]
):
    with Mesh((Topology("cta", 2560),), layout=(32, 80), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(128,), names=("d0",)) as scope:
            T.fill(out, 0.0)
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        wr = T.alloc_tensor(
            tensor_type=Tensor[
                (4, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 32, 4), (1, 0, 4)),
                    attrs=(B(), S(1)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        acc_3 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        xc = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        sv = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        with Mesh(
            (Topology("cta", 2560),), layout=((32, 32), (80, 1)), names=("d0", "d1")
        ) as scope_1:
            with Mesh(
                (Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")
            ) as scope_2:
                value_3 = T.tensor_view(
                    T.ptr_of(proj[0:0 + 1, (scope_1.d1 + 0) * 128:(scope_1.d1 + 0) * 128 + 1]),
                    layout=((256, 128), (16640, 1)),
                    shape=(256, 128),
                )
                value_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(11, 128), shape=(11, 128)
                )
                value_4 = T.tensor_view(T.ptr_of(value_2), layout=(11, 128), shape=(11, 128))
                value_5 = T.tensor_view(
                    T.ptr_of(rowidx[scope_1.d0 * 11:scope_1.d0 * 11 + 1]),
                    layout=(11,),
                    shape=(11,),
                )
                T.copy_async(value_3, value_4, value_5, fill=0.0)
                value_6 = T.tensor_view(T.ptr_of(value_2), layout=(8, 128), shape=(8, 128))
                value_7 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_6, value_7)
                value_8 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_7, value_8, dtype="f32")
                value_9 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 128), "bf16", ((4, 32 @ scope_2.d1, 4), (1, 0, 4)), "rmem"
                    ]
                )
                value_10 = T.tensor_view(
                    T.ptr_of(conv_w[(scope_1.d1 + 0) * 128:(scope_1.d1 + 0) * 128 + 1, 0:0 + 1]),
                    layout=((4, 128), (1, 4)),
                    shape=(4, 128),
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value_9),
                    layout=((4, 32 @ scope_2.d1, 4), (1, 0, 4)),
                    shape=(4, 128),
                )
                T.copy(value_10, value_11)
                T.cast(value_11, wr, dtype="f32")
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_12 = T.tensor_view(
                    T.ptr_of(wr), layout=((1, 32 @ scope_2.d1, 4), (1, 0, 4)), shape=(1, 128)
                )
                T.binary(value_8, value_12, acc, kind=BinaryKind.MUL)
                value_14 = T.tensor_view(
                    T.ptr_of(value_2[1:1 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_15 = T.tensor_view(
                    T.ptr_of(value_13),
                    layout=((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_14, value_15)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_15, value_16, dtype="f32")
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_18 = T.tensor_view(
                    T.ptr_of(wr[1:1 + 1, 0:0 + 128]), layout=((1, 128), (0, 4)), shape=(1, 128)
                )
                T.binary(value_16, value_18, value_17, kind=BinaryKind.MUL)
                acc_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(acc, value_17, acc_1, kind=BinaryKind.ADD)
                value_20 = T.tensor_view(
                    T.ptr_of(value_2[2:2 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_19),
                    layout=((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_20, value_21)
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_21, value_22, dtype="f32")
                value_23 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_24 = T.tensor_view(
                    T.ptr_of(wr[2:2 + 1, 0:0 + 128]), layout=((1, 128), (0, 4)), shape=(1, 128)
                )
                T.binary(value_22, value_24, value_23, kind=BinaryKind.MUL)
                acc_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(acc_1, value_23, acc_2, kind=BinaryKind.ADD)
                value_26 = T.tensor_view(
                    T.ptr_of(value_2[3:3 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_27 = T.tensor_view(
                    T.ptr_of(value_25),
                    layout=((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_26, value_27)
                value_28 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_27, value_28, dtype="f32")
                value_29 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_30 = T.tensor_view(
                    T.ptr_of(wr[3:3 + 1, 0:0 + 128]), layout=((1, 128), (0, 4)), shape=(1, 128)
                )
                T.binary(value_28, value_30, value_29, kind=BinaryKind.MUL)
                T.binary(acc_2, value_29, acc_3, kind=BinaryKind.ADD)
                value_31 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "bf16",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(acc_3, value_31, dtype="bf16")
                T.cast(value_31, xc, dtype="f32")
                value_32 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.unary(xc, value_32, kind=UnaryKind.NEG)
                value_33 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_32, value_33, kind=UnaryKind.EXP)
                value_34 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(value_33, scalar, value_34, kind=BinaryKind.ADD)
                value_35 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xc, value_34, value_35, kind=BinaryKind.DIV)
                T.cast(value_35, sv, dtype="bf16")
                sf = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(sv, sf, dtype="f32")
                value_36 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(sf, sf, value_36, kind=BinaryKind.MUL)
                value_37 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "bf16",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_36, value_37, dtype="bf16")
                sq = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_37, sq, dtype="f32")
                value_38 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.reduce(sq, value_38, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                ssum = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "bf16", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(value_38, ssum, dtype="bf16")
                value_39 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(ssum, value_39, dtype="f32")
                value_40 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 9.98377799987793e-07)
                T.binary(value_39, scalar_1, value_40, kind=BinaryKind.ADD)
                se = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "bf16", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(value_40, se, dtype="bf16")
                value_41 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(se, value_41, dtype="f32")
                value_42 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.unary(value_41, value_42, kind=UnaryKind.RSQRT)
                value_43 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "bf16", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(value_42, value_43, dtype="bf16")
                rr = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 1), "f32", ((4 @ scope_2.d0, 2, 1, 1), (2, 1, 0, 0)), "rmem"
                    ]
                )
                T.cast(value_43, rr, dtype="f32")
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(sf, rr, value_44, kind=BinaryKind.MUL)
                value_45 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "bf16",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_44, value_45, dtype="bf16")
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_2.d0, 2, 32 @ scope_2.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_45, y, dtype="f32")
                value_46 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 1024]), layout=(8, 128), shape=(8, 128)
                )
                value_47 = T.tensor_view(T.ptr_of(value_46), layout=(8, 128), shape=(8, 128))
                T.copy(y, value_47)
                value_48 = T.tensor_view(T.ptr_of(out), layout=(259, 10336), shape=(259, 10336))
                window = T.tensor_view(
                    T.ptr_of(value_48[scope_1.d0 * 8:scope_1.d0 * 8 + 8, (scope_1.d1 + 0) * 128:(scope_1.d1 + 0) * 128 + 128]),
                    layout=((8, 128), (10336, 1)),
                    shape=(8, 128),
                )
                T.copy(value_47, window)
        value_50 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        wr_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (4, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 32, 4), (1, 0, 4)),
                    attrs=(B(), S(1)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_62 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_68 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        value_74 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        acc_7 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        xc_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "f32",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        sv_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 128),
                "bf16",
                ShardLayout(
                    layout=Layout((4, 2, 32, 4), (0, 4, 0, 1)),
                    attrs=(S(0), S(2)),
                    mesh=Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        with cta[:, 32:] as scope_3:
            with Mesh(
                (Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")
            ) as scope_4:
                value_52 = T.tensor_view(
                    T.ptr_of(proj[0:0 + 1, (scope_3.d1 + 32) * 128:(scope_3.d1 + 32) * 128 + 1]),
                    layout=((256, 128), (16640, 1)),
                    shape=(256, 128),
                )
                value_51 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(11, 128), shape=(11, 128)
                )
                value_53 = T.tensor_view(T.ptr_of(value_51), layout=(11, 128), shape=(11, 128))
                value_54 = T.tensor_view(
                    T.ptr_of(rowidx[scope_3.d0 * 11:scope_3.d0 * 11 + 1]),
                    layout=(11,),
                    shape=(11,),
                )
                T.copy_async(value_52, value_53, value_54, fill=0.0)
                value_55 = T.tensor_view(T.ptr_of(value_51), layout=(8, 128), shape=(8, 128))
                value_56 = T.tensor_view(
                    T.ptr_of(value_50),
                    layout=((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_55, value_56)
                value_57 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_56, value_57, dtype="f32")
                value_58 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 128), "bf16", ((4, 32 @ scope_4.d1, 4), (1, 0, 4)), "rmem"
                    ]
                )
                value_59 = T.tensor_view(
                    T.ptr_of(conv_w[(scope_3.d1 + 32) * 128:(scope_3.d1 + 32) * 128 + 1, 0:0 + 1]),
                    layout=((4, 128), (1, 4)),
                    shape=(4, 128),
                )
                value_60 = T.tensor_view(
                    T.ptr_of(value_58),
                    layout=((4, 32 @ scope_4.d1, 4), (1, 0, 4)),
                    shape=(4, 128),
                )
                T.copy(value_59, value_60)
                T.cast(value_60, wr_1, dtype="f32")
                acc_4 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_61 = T.tensor_view(
                    T.ptr_of(wr_1), layout=((1, 32 @ scope_4.d1, 4), (1, 0, 4)), shape=(1, 128)
                )
                T.binary(value_57, value_61, acc_4, kind=BinaryKind.MUL)
                value_63 = T.tensor_view(
                    T.ptr_of(value_51[1:1 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_64 = T.tensor_view(
                    T.ptr_of(value_62),
                    layout=((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_63, value_64)
                value_65 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_64, value_65, dtype="f32")
                value_66 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_67 = T.tensor_view(
                    T.ptr_of(wr_1[1:1 + 1, 0:0 + 128]),
                    layout=((1, 128), (0, 4)),
                    shape=(1, 128),
                )
                T.binary(value_65, value_67, value_66, kind=BinaryKind.MUL)
                acc_5 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(acc_4, value_66, acc_5, kind=BinaryKind.ADD)
                value_69 = T.tensor_view(
                    T.ptr_of(value_51[2:2 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_70 = T.tensor_view(
                    T.ptr_of(value_68),
                    layout=((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_69, value_70)
                value_71 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_70, value_71, dtype="f32")
                value_72 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_73 = T.tensor_view(
                    T.ptr_of(wr_1[2:2 + 1, 0:0 + 128]),
                    layout=((1, 128), (0, 4)),
                    shape=(1, 128),
                )
                T.binary(value_71, value_73, value_72, kind=BinaryKind.MUL)
                acc_6 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(acc_5, value_72, acc_6, kind=BinaryKind.ADD)
                value_75 = T.tensor_view(
                    T.ptr_of(value_51[3:3 + 1, 0:0 + 1]), layout=(8, 128), shape=(8, 128)
                )
                value_76 = T.tensor_view(
                    T.ptr_of(value_74),
                    layout=((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                    shape=(8, 128),
                )
                T.copy(value_75, value_76)
                value_77 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_76, value_77, dtype="f32")
                value_78 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                value_79 = T.tensor_view(
                    T.ptr_of(wr_1[3:3 + 1, 0:0 + 128]),
                    layout=((1, 128), (0, 4)),
                    shape=(1, 128),
                )
                T.binary(value_77, value_79, value_78, kind=BinaryKind.MUL)
                T.binary(acc_6, value_78, acc_7, kind=BinaryKind.ADD)
                value_80 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "bf16",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(acc_7, value_80, dtype="bf16")
                T.cast(value_80, xc_1, dtype="f32")
                value_81 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.unary(xc_1, value_81, kind=UnaryKind.NEG)
                value_82 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.unary(value_81, value_82, kind=UnaryKind.EXP)
                value_83 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1.0)
                T.binary(value_82, scalar_2, value_83, kind=BinaryKind.ADD)
                value_84 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.binary(xc_1, value_83, value_84, kind=BinaryKind.DIV)
                T.cast(value_84, sv_1, dtype="bf16")
                value_85 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 128),
                        "f32",
                        ((4 @ scope_4.d0, 2, 32 @ scope_4.d1, 4), (0, 4, 0, 1)),
                        "rmem",
                    ]
                )
                T.cast(sv_1, value_85, dtype="f32")
                value_49_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 1024]), layout=(8, 128), shape=(8, 128)
                )
                value_86 = T.tensor_view(T.ptr_of(value_49_1), layout=(8, 128), shape=(8, 128))
                T.copy(value_85, value_86)
                value_87 = T.tensor_view(T.ptr_of(out), layout=(259, 10336), shape=(259, 10336))
                window_1 = T.tensor_view(
                    T.ptr_of(value_87[scope_3.d0 * 8:scope_3.d0 * 8 + 8, (scope_3.d1 + 32) * 128:(scope_3.d1 + 32) * 128 + 128]),
                    layout=((8, 128), (10336, 1)),
                    shape=(8, 128),
                )
                T.copy(value_86, window_1)
            with Mesh((Topology("thread", 128),), layout=(8,), names=("d0",)) as scope_5:
                value_89 = T.alloc_tensor(tensor_type=Tensor[(1,), "bf16", (1,), "rmem"])
                value_90 = T.tensor_view(
                    T.ptr_of(a_log[scope_3.d1:scope_3.d1 + 1]), layout=((1,), (0,)), shape=(1,)
                )
                value_91 = T.tensor_view(T.ptr_of(value_89), layout=(1,), shape=(1,))
                T.copy(value_90, value_91)
                value_92 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                T.cast(value_91, value_92, dtype="f32")
                value_93 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                T.unary(value_92, value_93, kind=UnaryKind.EXP)
                value_94 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                T.unary(value_93, value_94, kind=UnaryKind.NEG)
                value_95 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "bf16", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                value_96 = T.tensor_view(
                    T.ptr_of(proj[scope_3.d0 * 8:scope_3.d0 * 8 + 1, 16432 + scope_3.d1:16432 + scope_3.d1 + 1]),
                    layout=((8, 1), (16640, 0)),
                    shape=(8, 1),
                )
                value_97 = T.tensor_view(
                    T.ptr_of(value_95), layout=((8 @ scope_5.d0, 1), (0, 0)), shape=(8, 1)
                )
                T.copy(value_96, value_97)
                value_98 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.cast(value_97, value_98, dtype="f32")
                value_99 = T.alloc_tensor(tensor_type=Tensor[(1,), "bf16", (1,), "rmem"])
                value_100 = T.tensor_view(
                    T.ptr_of(dt_bias[scope_3.d1:scope_3.d1 + 1]),
                    layout=((1,), (0,)),
                    shape=(1,),
                )
                value_101 = T.tensor_view(T.ptr_of(value_99), layout=(1,), shape=(1,))
                T.copy(value_100, value_101)
                value_102 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
                T.cast(value_101, value_102, dtype="f32")
                av = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.binary(value_98, value_102, av, kind=BinaryKind.ADD)
                value_103 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "bool", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 20.0)
                T.binary(av, scalar_3, value_103, kind=BinaryKind.GT)
                value_104 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.unary(av, value_104, kind=UnaryKind.EXP)
                value_105 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_4, 1.0)
                T.binary(value_104, scalar_4, value_105, kind=BinaryKind.ADD)
                value_106 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.unary(value_105, value_106, kind=UnaryKind.LOG)
                sp = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.where(value_103, av, value_106, sp)
                gv = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.binary(value_94, sp, gv, kind=BinaryKind.MUL)
                value_88_1 = T.tensor_view(
                    T.ptr_of(scratch_2[0:0 + 8]), layout=(8, 1), shape=(8, 1)
                )
                value_107 = T.tensor_view(T.ptr_of(value_88_1), layout=(8, 1), shape=(8, 1))
                T.copy(gv, value_107)
                value_108 = T.tensor_view(
                    T.ptr_of(out), layout=(259, 10336), shape=(259, 10336)
                )
                window_2 = T.tensor_view(
                    T.ptr_of(value_108[scope_3.d0 * 8:scope_3.d0 * 8 + 8, 10240 + scope_3.d1:10240 + scope_3.d1 + 1]),
                    layout=((8, 1), (10336, 1)),
                    shape=(8, 1),
                )
                T.copy(value_107, window_2)
                value_110 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "bf16", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                value_111 = T.tensor_view(
                    T.ptr_of(proj[scope_3.d0 * 8:scope_3.d0 * 8 + 1, 16384 + scope_3.d1:16384 + scope_3.d1 + 1]),
                    layout=((8, 1), (16640, 0)),
                    shape=(8, 1),
                )
                value_112 = T.tensor_view(
                    T.ptr_of(value_110), layout=((8 @ scope_5.d0, 1), (0, 0)), shape=(8, 1)
                )
                T.copy(value_111, value_112)
                bv = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.cast(value_112, bv, dtype="f32")
                value_113 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.unary(bv, value_113, kind=UnaryKind.NEG)
                value_114 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.unary(value_113, value_114, kind=UnaryKind.EXP)
                value_115 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_5, 1.0)
                T.binary(value_114, scalar_5, value_115, kind=BinaryKind.ADD)
                value_116 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_6, 1.0)
                T.binary(scalar_6, value_115, value_116, kind=BinaryKind.DIV)
                value_117 = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "bf16", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.cast(value_116, value_117, dtype="bf16")
                be = T.alloc_tensor(
                    tensor_type=Tensor[(8, 1), "f32", ((8 @ scope_5.d0, 1), (0, 0)), "rmem"]
                )
                T.cast(value_117, be, dtype="f32")
                value_109_1 = T.tensor_view(
                    T.ptr_of(scratch_3[0:0 + 8]), layout=(8, 1), shape=(8, 1)
                )
                value_118 = T.tensor_view(T.ptr_of(value_109_1), layout=(8, 1), shape=(8, 1))
                T.copy(be, value_118)
                value_119 = T.tensor_view(
                    T.ptr_of(out), layout=(259, 10336), shape=(259, 10336)
                )
                window_3 = T.tensor_view(
                    T.ptr_of(value_119[scope_3.d0 * 8:scope_3.d0 * 8 + 8, 10288 + scope_3.d1:10288 + scope_3.d1 + 1]),
                    layout=((8, 1), (10336, 1)),
                    shape=(8, 1),
                )
                T.copy(value_118, window_3)
        with cta[31:] as scope_6:
            with Mesh((Topology("thread", 128),), layout=(128,), names=("d0",)) as scope_7:
                value_121 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (3, 128), "bf16", ((3, 128 @ scope_7.d0), (1, 0)), "rmem"
                    ]
                )
                value_122 = T.tensor_view(
                    T.ptr_of(proj[253:253 + 1, scope_6.d1 * 128:scope_6.d1 * 128 + 1]),
                    layout=((3, 128), (16640, 1)),
                    shape=(3, 128),
                )
                value_123 = T.tensor_view(
                    T.ptr_of(value_121), layout=((3, 128 @ scope_7.d0), (1, 0)), shape=(3, 128)
                )
                T.copy(value_122, value_123)
                value_124 = T.alloc_tensor(
                    tensor_type=Tensor[(3, 128), "f32", ((3, 128 @ scope_7.d0), (1, 0)), "rmem"]
                )
                T.cast(value_123, value_124, dtype="f32")
                value_120_1 = T.tensor_view(
                    T.ptr_of(scratch_4[0:0 + 384]), layout=(3, 128), shape=(3, 128)
                )
                value_125 = T.tensor_view(
                    T.ptr_of(value_120_1), layout=(3, 128), shape=(3, 128)
                )
                T.copy(value_124, value_125)
                value_126 = T.tensor_view(
                    T.ptr_of(out), layout=(259, 10336), shape=(259, 10336)
                )
                window_4 = T.tensor_view(
                    T.ptr_of(value_126[256:256 + 3, scope_6.d1 * 128:scope_6.d1 * 128 + 128]),
                    layout=((3, 128), (10336, 1)),
                    shape=(3, 128),
                )
                T.copy(value_125, window_4)
