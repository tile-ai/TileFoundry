# analysis target=nvidia.h200_sxm module=QKV function=whole topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r144.81MB/w35.00MB@logical,r27.98GB/w4.51GB@total,r217.05MB/w35.00MB@cta,r213.52MB/w35.00MB@thread;rmem:r573.36MB/w565.85MB@logical,r93.70GB/w89.62GB@total,r726.90MB/w695.24MB@cta,r3.02MB/w2.74MB@thread;smem:r213.66MB/w144.91MB@logical,r28.65GB/w28.65GB@total,r222.25MB/w222.25MB@cta,r39.63MB/w210.05MB@thread footprint=cos:0;gamma_k:0;gamma_q:0;sin:0;v192:570:0;v194:571:0;w:0;x:0 footprint-precision=lower_bound peak=gmem:178.31MB;rmem:384.00KB;smem:208.00KB persistent=gmem:150.25MB
#   error="rmem placement peak 384.00KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def whole(
    x: Tensor[(1, 1024, 5120), "bf16"], w: Tensor[(14336, 5120), "bf16"], gamma_q: ConstTensor[(256,), "bf16"], gamma_k: ConstTensor[(256,), "bf16"], cos: Tensor[(1024, 64), "bf16"], sin: Tensor[(1024, 64), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(1024, 14336), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            value_3_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for g in range(0, 0, 1):
                for bn in range(0, 56, 1):
                    for mi in range(((cta.d0 - (52 * g)) - (8 * bn)) - (132 * (((cta.d0 - (52 * g)) - (8 * bn)) // 132)), 8, 132):
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
                                value_1 = T.tensor_view(
                                    T.ptr_of(x[0:0 + 1, ((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, k:k + 1]),
                                    layout=((128, 64), (5120, 1)),
                                    shape=(128, 64),
                                )
                                value_2 = T.tensor_view(
                                    T.ptr_of(value_stages[stage_counter % 3]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                T.copy_async_tensor(value_1, value_2)
                                value_4 = T.tensor_view(
                                    T.ptr_of(w[bn * 256:bn * 256 + 1, k:k + 1]),
                                    layout=((256, 64), (5120, 1)),
                                    shape=(256, 64),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(value_3_stages[stage_counter % 3]),
                                    layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                T.copy_async_tensor(value_4, value_5)
                            with Mesh(
                                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                            ) as scope_3:
                                for o_m in range(0, 128, 128):
                                    for o_n in range(0, 256, 256):
                                        for o_k in range(0, 64, 64):
                                            value_6 = T.tensor_view(
                                                T.ptr_of(acc),
                                                layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                                shape=(128, 256),
                                            )
                                            value_7 = T.tensor_view(
                                                T.ptr_of(value_stages[stage_counter % 3]),
                                                layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                                shape=(128, 64),
                                            )
                                            lhs_view = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            value_8 = T.tensor_view(
                                                T.ptr_of(value_3_stages[stage_counter % 3]),
                                                layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                                shape=(64, 256),
                                            )
                                            rhs_view = T.tensor_view(
                                                T.ptr_of(value_8[o_k:o_k + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view,
                                                rhs_view,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_1 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_1,
                                                rhs_view_1,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_2 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_2,
                                                rhs_view_2,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                                            lhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_7[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                                layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(128, 16),
                                            )
                                            rhs_view_3 = T.tensor_view(
                                                T.ptr_of(value_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                                layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                value_6,
                                                lhs_view_3,
                                                rhs_view_3,
                                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                            )
                            T.binary(
                                stage_counter,
                                stage_one,
                                stage_counter,
                                kind=BinaryKind.ADD,
                            )
                        with Mesh(
                            (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_4:
                            value_10 = T.alloc_tensor(
                                tensor_type=Tensor[
                                    (128, 256),
                                    "bf16",
                                    ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    "rmem",
                                ]
                            )
                            value_11 = T.tensor_view(
                                T.ptr_of(acc),
                                layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                shape=(128, 256),
                            )
                            T.cast(value_11, value_10, dtype="bf16")
                            value_9 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_12 = T.tensor_view(
                                T.ptr_of(value_9),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            T.copy(value_10, value_12)
                        for _q in range(bn - (56 * (bn // 56)), 48, 1048576):
                            for _h in range(bn - (2 * (bn // 2)), 1, 2):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                                ) as scope_5:
                                    value_13 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_9_2 = T.tensor_view(
                                        147456,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_14 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_15 = T.tensor_view(
                                        T.ptr_of(value_13),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_14, value_15)
                                    x32 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_15, x32, dtype="f32")
                                    value_16 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_17 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_18 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, x32, value_18, kind=BinaryKind.MUL)
                                    value_19 = T.tensor_view(
                                        T.ptr_of(value_17),
                                        layout=((16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_18, value_19)
                                    value_20 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_5.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    T.reduce(
                                        value_19,
                                        value_20,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_21 = T.tensor_view(
                                        T.ptr_of(value_16),
                                        layout=(16 @ scope_5.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_20, value_21)
                                    value_22 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    scalar = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar, 1e-06)
                                    T.binary(value_21, scalar, value_22, kind=BinaryKind.ADD)
                                    value_23 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_22, value_23, kind=UnaryKind.RSQRT)
                                    value_24 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32, value_23, value_24, kind=BinaryKind.MUL)
                                    value_25 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_26 = T.tensor_view(
                                        T.ptr_of(gamma_q),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_27 = T.tensor_view(
                                        T.ptr_of(value_25),
                                        layout=(1, 1, 16 @ scope_5.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_26, value_27)
                                    value_28 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_27, value_28, dtype="f32")
                                    gw = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    scalar_1 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_1, 1.0)
                                    T.binary(value_28, scalar_1, gw, kind=BinaryKind.ADD)
                                    value_29 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_24, gw, value_29, kind=BinaryKind.MUL)
                                    nrm = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_29, nrm, dtype="bf16")
                                    value_30 = T.tensor_view(
                                        T.ptr_of(nrm),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 16),
                                        shape=(128, 256),
                                    )
                                    value_31 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_30, value_31)
                                    value_32 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_33 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_34 = T.tensor_view(
                                        T.ptr_of(value_32),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_33, value_34)
                                    lo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_34, lo, dtype="f32")
                                    value_35 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_36 = T.tensor_view(
                                        T.ptr_of(cos[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_37 = T.tensor_view(
                                        T.ptr_of(value_35),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_36, value_37)
                                    cl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_37, cl, dtype="f32")
                                    value_38 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, cl, value_38, kind=BinaryKind.MUL)
                                    value_39 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_38, value_39, dtype="bf16")
                                    value_40 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_39, value_40, dtype="f32")
                                    value_41 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_42 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_43 = T.tensor_view(
                                        T.ptr_of(value_41),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_42, value_43)
                                    hi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_43, hi, dtype="f32")
                                    value_44 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi, value_44, kind=UnaryKind.NEG)
                                    value_45 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_46 = T.tensor_view(
                                        T.ptr_of(sin[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_47 = T.tensor_view(
                                        T.ptr_of(value_45),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_46, value_47)
                                    sl = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_47, sl, dtype="f32")
                                    value_48 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_44, sl, value_48, kind=BinaryKind.MUL)
                                    value_49 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_48, value_49, dtype="bf16")
                                    value_50 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_49, value_50, dtype="f32")
                                    value_51 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_40, value_50, value_51, kind=BinaryKind.ADD)
                                    rlo = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_51, rlo, dtype="bf16")
                                    value_52 = T.tensor_view(
                                        T.ptr_of(rlo),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(128, 32),
                                    )
                                    value_53 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window = T.tensor_view(
                                        T.ptr_of(value_53[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_52, window)
                                    value_54 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_55 = T.tensor_view(
                                        T.ptr_of(cos[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_56 = T.tensor_view(
                                        T.ptr_of(value_54),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_55, value_56)
                                    ch = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_56, ch, dtype="f32")
                                    value_57 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi, ch, value_57, kind=BinaryKind.MUL)
                                    value_58 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_57, value_58, dtype="bf16")
                                    value_59 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_58, value_59, dtype="f32")
                                    value_60 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_61 = T.tensor_view(
                                        T.ptr_of(sin[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_62 = T.tensor_view(
                                        T.ptr_of(value_60),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_61, value_62)
                                    sh = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_62, sh, dtype="f32")
                                    value_63 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo, sh, value_63, kind=BinaryKind.MUL)
                                    value_64 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_63, value_64, dtype="bf16")
                                    value_65 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_64, value_65, dtype="f32")
                                    value_66 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_59, value_65, value_66, kind=BinaryKind.ADD)
                                    rhi = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_66, rhi, dtype="bf16")
                                    value_67 = T.tensor_view(
                                        T.ptr_of(rhi),
                                        layout=(16 @ scope_5.d0, 8, 16 @ scope_5.d1, 2),
                                        shape=(128, 32),
                                    )
                                    value_68 = T.tensor_view(
                                        T.ptr_of(value_9_2),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_1 = T.tensor_view(
                                        T.ptr_of(value_68[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_67, window_1)
                        for _k0 in range(47, bn - (56 * (bn // 56)), 1048576):
                            for _k1 in range(bn - (56 * (bn // 56)), 52, 1048576):
                                with Mesh(
                                    (Topology("thread", 384),), layout=(16, 16), names=("d0", "d1")
                                ) as scope_6:
                                    value_69 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_9_3 = T.tensor_view(
                                        147456,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    value_70 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=Layout((16, 8, (4, 4), 16), (512, 64, (8192, 16), 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 16),
                                    )
                                    value_71 = T.tensor_view(
                                        T.ptr_of(value_69),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_70, value_71)
                                    x32_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_71, x32_1, dtype="f32")
                                    value_72 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    value_73 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            ((16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16), (0, 16, 0, 1)),
                                            "rmem",
                                        ]
                                    )
                                    value_74 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, x32_1, value_74, kind=BinaryKind.MUL)
                                    value_75 = T.tensor_view(
                                        T.ptr_of(value_73),
                                        layout=((16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16), (0, 16, 0, 1)),
                                        shape=(16, 8, 16, 16),
                                    )
                                    T.copy(value_74, value_75)
                                    value_76 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            ((16 @ scope_6.d0, 8, 1, 1, 1), (8, 1, 0, 0, 0)),
                                            "rmem",
                                        ]
                                    )
                                    T.reduce(
                                        value_75,
                                        value_76,
                                        axes=(2, 3),
                                        keepdim=True,
                                        kind=ReduceKind.MEAN,
                                    )
                                    value_77 = T.tensor_view(
                                        T.ptr_of(value_72),
                                        layout=(16 @ scope_6.d0, 8, 1, 1),
                                        shape=(16, 8, 1, 1),
                                    )
                                    T.copy(value_76, value_77)
                                    value_78 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    scalar_2 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_2, 1e-06)
                                    T.binary(value_77, scalar_2, value_78, kind=BinaryKind.ADD)
                                    value_79 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 1, 1),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 1, 1),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(value_78, value_79, kind=UnaryKind.RSQRT)
                                    value_80 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(x32_1, value_79, value_80, kind=BinaryKind.MUL)
                                    value_81 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "bf16",
                                            (1, 1, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    value_82 = T.tensor_view(
                                        T.ptr_of(gamma_k),
                                        layout=((1, 1, 16, 16), (0, 0, 16, 1)),
                                        shape=(1, 1, 16, 16),
                                    )
                                    value_83 = T.tensor_view(
                                        T.ptr_of(value_81),
                                        layout=(1, 1, 16 @ scope_6.d1, 16),
                                        shape=(1, 1, 16, 16),
                                    )
                                    T.copy(value_82, value_83)
                                    value_84 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_83, value_84, dtype="f32")
                                    gw_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (1, 1, 16, 16),
                                            "f32",
                                            (1, 1, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    scalar_3 = T.alloc_tensor(
                                        tensor_type=Tensor[(), "f32", "rmem"]
                                    )
                                    T.fill(scalar_3, 1.0)
                                    T.binary(value_84, scalar_3, gw_1, kind=BinaryKind.ADD)
                                    value_85 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_80, gw_1, value_85, kind=BinaryKind.MUL)
                                    nrm_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 16),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_85, nrm_1, dtype="bf16")
                                    value_86 = T.tensor_view(
                                        T.ptr_of(nrm_1),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 16),
                                        shape=(128, 256),
                                    )
                                    value_87 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    T.copy(value_86, value_87)
                                    value_88 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_89 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_90 = T.tensor_view(
                                        T.ptr_of(value_88),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_89, value_90)
                                    lo_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_90, lo_1, dtype="f32")
                                    value_91 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_92 = T.tensor_view(
                                        T.ptr_of(cos[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_93 = T.tensor_view(
                                        T.ptr_of(value_91),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_92, value_93)
                                    cl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_93, cl_1, dtype="f32")
                                    value_94 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, cl_1, value_94, kind=BinaryKind.MUL)
                                    value_95 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_94, value_95, dtype="bf16")
                                    value_96 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_95, value_96, dtype="f32")
                                    value_97 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_98 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)) + 32 | Swizzle(3, 4, 3),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_99 = T.tensor_view(
                                        T.ptr_of(value_97),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_98, value_99)
                                    hi_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_99, hi_1, dtype="f32")
                                    value_100 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.unary(hi_1, value_100, kind=UnaryKind.NEG)
                                    value_101 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_102 = T.tensor_view(
                                        T.ptr_of(sin[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 0:0 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_103 = T.tensor_view(
                                        T.ptr_of(value_101),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_102, value_103)
                                    sl_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_103, sl_1, dtype="f32")
                                    value_104 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(value_100, sl_1, value_104, kind=BinaryKind.MUL)
                                    value_105 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_104, value_105, dtype="bf16")
                                    value_106 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_105, value_106, dtype="f32")
                                    value_107 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_96,
                                        value_106,
                                        value_107,
                                        kind=BinaryKind.ADD,
                                    )
                                    rlo_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_107, rlo_1, dtype="bf16")
                                    value_108 = T.tensor_view(
                                        T.ptr_of(rlo_1),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(128, 32),
                                    )
                                    value_109 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_2 = T.tensor_view(
                                        T.ptr_of(value_109[0:0 + 128, 0:0 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_108, window_2)
                                    value_110 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_111 = T.tensor_view(
                                        T.ptr_of(cos[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_112 = T.tensor_view(
                                        T.ptr_of(value_110),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_111, value_112)
                                    ch_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_112, ch_1, dtype="f32")
                                    value_113 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(hi_1, ch_1, value_113, kind=BinaryKind.MUL)
                                    value_114 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_113, value_114, dtype="bf16")
                                    value_115 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_114, value_115, dtype="f32")
                                    value_116 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    value_117 = T.tensor_view(
                                        T.ptr_of(sin[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 1, 32:32 + 1]),
                                        layout=((16, 8, 16, 2), (512, 64, 2, 1)),
                                        shape=(16, 8, 16, 2),
                                    )
                                    value_118 = T.tensor_view(
                                        T.ptr_of(value_116),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(16, 8, 16, 2),
                                    )
                                    T.copy(value_117, value_118)
                                    sh_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_118, sh_1, dtype="f32")
                                    value_119 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(lo_1, sh_1, value_119, kind=BinaryKind.MUL)
                                    value_120 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_119, value_120, dtype="bf16")
                                    value_121 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_120, value_121, dtype="f32")
                                    value_122 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "f32",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.binary(
                                        value_115,
                                        value_121,
                                        value_122,
                                        kind=BinaryKind.ADD,
                                    )
                                    rhi_1 = T.alloc_tensor(
                                        tensor_type=Tensor[
                                            (16, 8, 16, 2),
                                            "bf16",
                                            (16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                            "rmem",
                                        ]
                                    )
                                    T.cast(value_122, rhi_1, dtype="bf16")
                                    value_123 = T.tensor_view(
                                        T.ptr_of(rhi_1),
                                        layout=(16 @ scope_6.d0, 8, 16 @ scope_6.d1, 2),
                                        shape=(128, 32),
                                    )
                                    value_124 = T.tensor_view(
                                        T.ptr_of(value_9_3),
                                        layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 256),
                                    )
                                    window_3 = T.tensor_view(
                                        T.ptr_of(value_124[0:0 + 128, 32:32 + 32]),
                                        layout=Layout((128, 32), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 32),
                                    )
                                    T.copy(value_123, window_3)
                        with scope[:1, :32] as scope_7:
                            value_9_4 = T.tensor_view(
                                147456,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_126 = T.tensor_view(
                                T.ptr_of(value_9_4),
                                layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                                shape=(128, 256),
                            )
                            value_125_1 = T.tensor_view(
                                T.ptr_of(scratch[0:0 + 32768]),
                                layout=(128, 256),
                                shape=(128, 256),
                            )
                            value_127 = T.tensor_view(
                                T.ptr_of(value_125_1), layout=(128, 256), shape=(128, 256)
                            )
                            T.copy_async_tensor(value_126, value_127)
                            value_128 = T.tensor_view(
                                T.ptr_of(out), layout=(1024, 14336), shape=(1024, 14336)
                            )
                            window_4 = T.tensor_view(
                                T.ptr_of(value_128[((g * 8) + mi) * 128:((g * 8) + mi) * 128 + 128, bn * 256:bn * 256 + 256]),
                                layout=((128, 256), (14336, 1)),
                                shape=(128, 256),
                            )
                            T.copy(value_127, window_4)
