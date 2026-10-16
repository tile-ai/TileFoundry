# analysis target=nvidia.h200_sxm module=QKV function=whole_last topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r139.69MB/w6.75MB@logical,r26.98GB/w891.00MB@total,r209.30MB/w6.75MB@cta,r205.89MB/w6.75MB@thread;rmem:r552.98MB/w545.73MB@logical,r90.36GB/w86.42GB@total,r700.94MB/w670.41MB@cta,r2.92MB/w2.65MB@thread;smem:r206.03MB/w139.78MB@logical,r27.63GB/w27.63GB@total,r214.31MB/w214.31MB@cta,r38.21MB/w202.55MB@thread footprint=cos:256.00KB;gamma_k:512B;gamma_q:512B;sin:256.00KB;v189:712:64.00KB;v191:713:8.25MB;w:288.00KB;x:256.00KB footprint-precision=exact peak=gmem:670.00MB;rmem:384.00KB;smem:208.00KB persistent=gmem:446.00MB
#   buffer=cos holds=51.31MB time=bn space=none reuse=13.25MB fits=no precision=exact
#   buffer=sin holds=51.31MB time=bn space=none reuse=13.25MB fits=no precision=exact
#   buffer=gamma_q holds=51.31MB time=bn space=cta.persistent reuse=3.48MB fits=no precision=exact
#   buffer=gamma_k holds=51.31MB time=bn space=cta.persistent reuse=3.48MB fits=no precision=exact
#   error="rmem placement peak 384.00KB exceeds capacity 256.00KB"
#   error="l2 reuse window bn holds 51.31MB at a 132-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole_last(
    out: Tensor[(8192, 14336), "bf16"], x: Tensor[(1, 8192, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(8192, 64), "bf16"], sin: Tensor[(8192, 64), "bf16"], out_1: Tensor[(8192, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(8192, 14336), shape=(8192, 14336))
            T.copy(value, out_1)
            value_1_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_4_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for bn in range(0, 54, 1):
                for mi in range(((cta.d0 - (16 * bn)) - (132 * (((cta.d0 - (16 * bn)) - 48) // 132))) - 48, 16, 132):
                    acc = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 256),
                            "f32",
                            ShardLayout(
                                layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                attrs=(S(0), S(1), S(3), S(5)),
                                mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                            ),
                            "rmem",
                        ]
                    )
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_1:
                        T.fill(acc, 0.0)
                    for k in range(0, 5120, 64):
                        with scope[2:, :32] as scope_2:
                            value_2 = T.tensor_view(
                                T.ptr_of(x[0:0 + 1, ((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, k:k + 1]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            value_3 = T.tensor_view(
                                T.ptr_of(value_1_stages[stage_counter % 3]),
                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(value_2, value_3)
                            value_5 = T.tensor_view(
                                T.ptr_of(w[bn * 256:bn * 256 + 1, k:k + 1]),
                                layout=((256, 64), (5120, 1)),
                                shape=(256, 64),
                            )
                            value_6 = T.tensor_view(
                                T.ptr_of(value_4_stages[stage_counter % 3]),
                                layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            T.copy_async_tensor(value_5, value_6)
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_3:
                            for o_m in range(0, 128, 128):
                                for o_n in range(0, 256, 256):
                                    for o_k in range(0, 64, 64):
                                        value_7 = T.tensor_view(
                                            T.ptr_of(acc),
                                            layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                            shape=(128, 256),
                                        )
                                        value_8 = T.tensor_view(
                                            T.ptr_of(value_1_stages[stage_counter % 3]),
                                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                            shape=(128, 64),
                                        )
                                        lhs_view = T.tensor_view(
                                            T.ptr_of(value_8[o_m:o_m + 128, o_k:o_k + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        value_9 = T.tensor_view(
                                            T.ptr_of(value_4_stages[stage_counter % 3]),
                                            layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                            shape=(64, 256),
                                        )
                                        rhs_view = T.tensor_view(
                                            T.ptr_of(value_9[o_k:o_k + 16, o_n:o_n + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_7,
                                            lhs_view,
                                            rhs_view,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_1 = T.tensor_view(
                                            T.ptr_of(value_8[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_1 = T.tensor_view(
                                            T.ptr_of(value_9[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_7,
                                            lhs_view_1,
                                            rhs_view_1,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_2 = T.tensor_view(
                                            T.ptr_of(value_8[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_2 = T.tensor_view(
                                            T.ptr_of(value_9[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_7,
                                            lhs_view_2,
                                            rhs_view_2,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                                        lhs_view_3 = T.tensor_view(
                                            T.ptr_of(value_8[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_3 = T.tensor_view(
                                            T.ptr_of(value_9[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                            layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            value_7,
                                            lhs_view_3,
                                            rhs_view_3,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                        T.binary(stage_counter, stage_one, stage_counter, kind=BinaryKind.ADD)
                    with Mesh(
                        (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_4:
                        value_11 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_12 = T.tensor_view(
                            T.ptr_of(acc),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 256),
                        )
                        T.cast(value_12, value_11, dtype="bf16")
                        value_10 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_13 = T.tensor_view(
                            T.ptr_of(value_10),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(value_11, value_13)
                    for _q in range(bn - (56 * (bn // 56)), 48, 1048576):
                        for _h in range(bn - (2 * (bn // 2)), 1, 2):
                            with Mesh(
                                (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                            ) as scope_5:
                                value_14 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                value_10_2 = T.tensor_view(
                                    147456,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                value_15 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 16),
                                )
                                value_16 = T.tensor_view(
                                    T.ptr_of(value_14),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                    shape=(16, 8, 16, 16),
                                )
                                T.copy(value_15, value_16)
                                x32 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_16, x32, dtype="f32")
                                value_17 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_5.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                value_18 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        ((16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16), (0, 16, 0, 1)),
                                        "rmem",
                                    ]
                                )
                                value_19 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(x32, x32, value_19, kind=BinaryKind.MUL)
                                value_20 = T.tensor_view(
                                    T.ptr_of(value_18),
                                    layout=((16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16), (0, 16, 0, 1)),
                                    shape=(16, 8, 16, 16),
                                )
                                T.copy(value_19, value_20)
                                value_21 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1),
                                        "f32",
                                        ((16 @ scope_5.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        "rmem",
                                    ]
                                )
                                T.reduce(
                                    value_20,
                                    value_21,
                                    axes=(2, 3),
                                    keepdim=True,
                                    kind=ReduceKind.MEAN,
                                )
                                value_22 = T.tensor_view(
                                    T.ptr_of(value_17),
                                    layout=(16 @ scope_5.d0, 8, 1, 1),
                                    shape=(16, 8, 1, 1),
                                )
                                T.copy(value_21, value_22)
                                value_23 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_5.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar, 1e-06)
                                T.binary(value_22, scalar, value_23, kind=BinaryKind.ADD)
                                value_24 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_5.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                T.unary(value_23, value_24, kind=UnaryKind.RSQRT)
                                value_25 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(x32, value_24, value_25, kind=BinaryKind.MUL)
                                value_26 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "bf16",
                                        (1, 1, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                value_27 = T.tensor_view(
                                    T.ptr_of(gamma_q),
                                    layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                    shape=(1, 1, 16, 16),
                                )
                                value_28 = T.tensor_view(
                                    T.ptr_of(value_26),
                                    layout=(1, 1, 16 @ scope_5.d1, 16),
                                    shape=(1, 1, 16, 16),
                                )
                                T.copy(value_27, value_28)
                                value_29 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "f32",
                                        (1, 1, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_28, value_29, dtype="f32")
                                gw = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "f32",
                                        (1, 1, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_1, 1.0)
                                T.binary(value_29, scalar_1, gw, kind=BinaryKind.ADD)
                                value_30 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_25, gw, value_30, kind=BinaryKind.MUL)
                                nrm = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_30, nrm, dtype="bf16")
                                value_31 = T.tensor_view(
                                    T.ptr_of(nrm),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                    shape=(128, 256),
                                )
                                value_32 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                T.copy(value_31, value_32)
                                value_33 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_34 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                value_35 = T.tensor_view(
                                    T.ptr_of(value_33),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_34, value_35)
                                lo = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_35, lo, dtype="f32")
                                value_36 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_37 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 0:0 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_38 = T.tensor_view(
                                    T.ptr_of(value_36),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_37, value_38)
                                cl = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_38, cl, dtype="f32")
                                value_39 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(lo, cl, value_39, kind=BinaryKind.MUL)
                                value_40 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_39, value_40, dtype="bf16")
                                value_41 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_40, value_41, dtype="f32")
                                value_42 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_43 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                value_44 = T.tensor_view(
                                    T.ptr_of(value_42),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_43, value_44)
                                hi = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_44, hi, dtype="f32")
                                value_45 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.unary(hi, value_45, kind=UnaryKind.NEG)
                                value_46 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_47 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 0:0 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_48 = T.tensor_view(
                                    T.ptr_of(value_46),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_47, value_48)
                                sl = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_48, sl, dtype="f32")
                                value_49 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_45, sl, value_49, kind=BinaryKind.MUL)
                                value_50 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_49, value_50, dtype="bf16")
                                value_51 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_50, value_51, dtype="f32")
                                value_52 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_41, value_51, value_52, kind=BinaryKind.ADD)
                                rlo = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_52, rlo, dtype="bf16")
                                value_53 = T.tensor_view(
                                    T.ptr_of(rlo),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(128, 32),
                                )
                                value_54 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                window = T.tensor_view(
                                    T.ptr_of(value_54[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(value_53, window)
                                value_55 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_56 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 32:32 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_57 = T.tensor_view(
                                    T.ptr_of(value_55),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_56, value_57)
                                ch = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_57, ch, dtype="f32")
                                value_58 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(hi, ch, value_58, kind=BinaryKind.MUL)
                                value_59 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_58, value_59, dtype="bf16")
                                value_60 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_59, value_60, dtype="f32")
                                value_61 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_62 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 32:32 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_63 = T.tensor_view(
                                    T.ptr_of(value_61),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_62, value_63)
                                sh = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_63, sh, dtype="f32")
                                value_64 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(lo, sh, value_64, kind=BinaryKind.MUL)
                                value_65 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_64, value_65, dtype="bf16")
                                value_66 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_65, value_66, dtype="f32")
                                value_67 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_60, value_66, value_67, kind=BinaryKind.ADD)
                                rhi = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_67, rhi, dtype="bf16")
                                value_68 = T.tensor_view(
                                    T.ptr_of(rhi),
                                    layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                    shape=(128, 32),
                                )
                                value_69 = T.tensor_view(
                                    T.ptr_of(value_10_2),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                window_1 = T.tensor_view(
                                    T.ptr_of(value_69[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(value_68, window_1)
                    for _k0 in range(47, bn - (56 * (bn // 56)), 1048576):
                        for _k1 in range(bn - (56 * (bn // 56)), 52, 1048576):
                            with Mesh(
                                (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                            ) as scope_6:
                                value_70 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                value_10_3 = T.tensor_view(
                                    147456,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                value_71 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 16),
                                )
                                value_72 = T.tensor_view(
                                    T.ptr_of(value_70),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                    shape=(16, 8, 16, 16),
                                )
                                T.copy(value_71, value_72)
                                x32_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_72, x32_1, dtype="f32")
                                value_73 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_6.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                value_74 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        ((16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16), (0, 16, 0, 1)),
                                        "rmem",
                                    ]
                                )
                                value_75 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(x32_1, x32_1, value_75, kind=BinaryKind.MUL)
                                value_76 = T.tensor_view(
                                    T.ptr_of(value_74),
                                    layout=((16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16), (0, 16, 0, 1)),
                                    shape=(16, 8, 16, 16),
                                )
                                T.copy(value_75, value_76)
                                value_77 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1),
                                        "f32",
                                        ((16 @ scope_6.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                        "rmem",
                                    ]
                                )
                                T.reduce(
                                    value_76,
                                    value_77,
                                    axes=(2, 3),
                                    keepdim=True,
                                    kind=ReduceKind.MEAN,
                                )
                                value_78 = T.tensor_view(
                                    T.ptr_of(value_73),
                                    layout=(16 @ scope_6.d0, 8, 1, 1),
                                    shape=(16, 8, 1, 1),
                                )
                                T.copy(value_77, value_78)
                                value_79 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_6.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_2, 1e-06)
                                T.binary(value_78, scalar_2, value_79, kind=BinaryKind.ADD)
                                value_80 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 1, 1), "f32", (16 @ scope_6.d0, 8, 1, 1), "rmem"
                                    ]
                                )
                                T.unary(value_79, value_80, kind=UnaryKind.RSQRT)
                                value_81 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(x32_1, value_80, value_81, kind=BinaryKind.MUL)
                                value_82 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "bf16",
                                        (1, 1, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                value_83 = T.tensor_view(
                                    T.ptr_of(gamma_k),
                                    layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                    shape=(1, 1, 16, 16),
                                )
                                value_84 = T.tensor_view(
                                    T.ptr_of(value_82),
                                    layout=(1, 1, 16 @ scope_6.d1, 16),
                                    shape=(1, 1, 16, 16),
                                )
                                T.copy(value_83, value_84)
                                value_85 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "f32",
                                        (1, 1, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_84, value_85, dtype="f32")
                                gw_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (1, 1, 16, 16),
                                        "f32",
                                        (1, 1, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                                T.fill(scalar_3, 1.0)
                                T.binary(value_85, scalar_3, gw_1, kind=BinaryKind.ADD)
                                value_86 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_81, gw_1, value_86, kind=BinaryKind.MUL)
                                nrm_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 16),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_86, nrm_1, dtype="bf16")
                                value_87 = T.tensor_view(
                                    T.ptr_of(nrm_1),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                    shape=(128, 256),
                                )
                                value_88 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                T.copy(value_87, value_88)
                                value_89 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_90 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                value_91 = T.tensor_view(
                                    T.ptr_of(value_89),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_90, value_91)
                                lo_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_91, lo_1, dtype="f32")
                                value_92 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_93 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 0:0 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_94 = T.tensor_view(
                                    T.ptr_of(value_92),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_93, value_94)
                                cl_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_94, cl_1, dtype="f32")
                                value_95 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(lo_1, cl_1, value_95, kind=BinaryKind.MUL)
                                value_96 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_95, value_96, dtype="bf16")
                                value_97 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_96, value_97, dtype="f32")
                                value_98 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_99 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                    shape=(16, 8, 16, 2),
                                )
                                value_100 = T.tensor_view(
                                    T.ptr_of(value_98),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_99, value_100)
                                hi_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_100, hi_1, dtype="f32")
                                value_101 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.unary(hi_1, value_101, kind=UnaryKind.NEG)
                                value_102 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_103 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 0:0 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_104 = T.tensor_view(
                                    T.ptr_of(value_102),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_103, value_104)
                                sl_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_104, sl_1, dtype="f32")
                                value_105 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_101, sl_1, value_105, kind=BinaryKind.MUL)
                                value_106 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_105, value_106, dtype="bf16")
                                value_107 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_106, value_107, dtype="f32")
                                value_108 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_97, value_107, value_108, kind=BinaryKind.ADD)
                                rlo_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_108, rlo_1, dtype="bf16")
                                value_109 = T.tensor_view(
                                    T.ptr_of(rlo_1),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(128, 32),
                                )
                                value_110 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                window_2 = T.tensor_view(
                                    T.ptr_of(value_110[0:0 + 128, 0:0 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(value_109, window_2)
                                value_111 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_112 = T.tensor_view(
                                    T.ptr_of(cos[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 32:32 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_113 = T.tensor_view(
                                    T.ptr_of(value_111),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_112, value_113)
                                ch_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_113, ch_1, dtype="f32")
                                value_114 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(hi_1, ch_1, value_114, kind=BinaryKind.MUL)
                                value_115 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_114, value_115, dtype="bf16")
                                value_116 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_115, value_116, dtype="f32")
                                value_117 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                value_118 = T.tensor_view(
                                    T.ptr_of(sin[((3 * 16) + mi) * 128:((3 * 16) + mi) * 128 + 1, 32:32 + 1]),
                                    layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                    shape=(16, 8, 16, 2),
                                )
                                value_119 = T.tensor_view(
                                    T.ptr_of(value_117),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(16, 8, 16, 2),
                                )
                                T.copy(value_118, value_119)
                                sh_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_119, sh_1, dtype="f32")
                                value_120 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(lo_1, sh_1, value_120, kind=BinaryKind.MUL)
                                value_121 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_120, value_121, dtype="bf16")
                                value_122 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_121, value_122, dtype="f32")
                                value_123 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "f32",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.binary(value_116, value_122, value_123, kind=BinaryKind.ADD)
                                rhi_1 = T.alloc_tensor(
                                    tensor_type=Tensor[
                                        (16, 8, 16, 2),
                                        "bf16",
                                        (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        "rmem",
                                    ]
                                )
                                T.cast(value_123, rhi_1, dtype="bf16")
                                value_124 = T.tensor_view(
                                    T.ptr_of(rhi_1),
                                    layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                    shape=(128, 32),
                                )
                                value_125 = T.tensor_view(
                                    T.ptr_of(value_10_3),
                                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                window_3 = T.tensor_view(
                                    T.ptr_of(value_125[0:0 + 128, 32:32 + 32]),
                                    layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 32),
                                )
                                T.copy(value_124, window_3)
                    with scope[:1, :32] as scope_7:
                        value_10_4 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_126 = T.tensor_view(
                            T.ptr_of(value_10_4),
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        value_127 = T.tensor_view(
                            T.ptr_of(out_1[(128 * mi) + 6144:(128 * mi) + 6144 + 1, 256 * bn:256 * bn + 1]),
                            layout=((128, 256), (14336, 1)),
                            shape=(128, 256),
                        )
                        T.copy_async_tensor(value_126, value_127)
