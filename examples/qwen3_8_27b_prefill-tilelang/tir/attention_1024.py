# analysis target=nvidia.h200_sxm module=Attention function=attention topology=cta wave=132/192
# selection requested=memory executed=memory
# memory traffic=gmem:r1.20MB/w12.12MB@logical,r231.86MB/w2.27GB@total,r1.21MB/w12.12MB@cta,r1.14MB/w12.12MB@thread;rmem:r10.47MB/w8.95MB@logical,r1.96GB/w1.77GB@total,r10.47MB/w9.42MB@cta,r44.44KB/w37.74KB@thread;smem:r1.89MB/w1.14MB@logical,r363.00MB/w219.00MB@total,r1.89MB/w1.14MB@cta,r239.50KB/w1.08MB@thread footprint=iota:512B;qkv:16.81MB;v120:88:64.00KB;v123:89:8.25MB footprint-precision=exact peak=gmem:40.06MB;rmem:320.02KB;smem:224.00KB persistent=gmem:28.00MB
#   buffer=qkv holds=25.13MB time=none space=cta.qb reuse=84.06MB fits=yes precision=exact
#   buffer=iota holds=320.31KB time=kb space=cta.qb,cta.h reuse=535.94KB fits=yes precision=exact
#   error="rmem placement peak 320.02KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attention(
    qkv: Tensor[(1024, 14336), "bf16"], iota: Tensor[(128,), "i32"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(1024, 6144), "bf16"]
):
    with Mesh((Topology("cta", 192),), layout=(8, 24), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            o = T.alloc_tensor(
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
            mx = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 1), (64, 16, 8, 1, 0)),
                        attrs=(S(0), S(1), S(3), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            rows = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 1),
                    "i32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 1), (64, 16, 8, 1, 0)),
                        attrs=(S(0), S(1), S(3), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            lsum = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 1), (64, 16, 8, 1, 0)),
                        attrs=(S(0), S(1), S(3), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_1:
                T.fill(o, 0.0)
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 1),
                        "f32",
                        ((2 @ scope_1.d0, 4 @ scope_1.d1, 2, 8 @ scope_1.d2, 1), (64, 16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.fill(value, 0.0)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1e+30)
                T.binary(value, scalar, mx, kind=BinaryKind.SUB)
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 1),
                        "i32",
                        ((2 @ scope_1.d0, 4 @ scope_1.d1, 2, 8 @ scope_1.d2, 1), (64, 16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                value_2 = T.tensor_view(
                    T.ptr_of(iota), layout=((128, 1), (1, 0)), shape=(128, 1)
                )
                value_3 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((2 @ scope_1.d0, 4 @ scope_1.d1, 2, 8 @ scope_1.d2, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.copy(value_2, value_3)
                value_4 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
                T.fill(value_4, 0.0)
                value_5 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar_1, ((8 - 1) - cta.d0) * 128)
                T.binary(value_4, scalar_1, value_5, kind=BinaryKind.ADD)
                value_6 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
                T.cast(value_5, value_6, dtype="i32")
                T.binary(value_3, value_6, rows, kind=BinaryKind.ADD)
                T.fill(lsum, 0.0)
            with scope[2:, :32] as scope_2:
                value_8 = T.tensor_view(
                    T.ptr_of(qkv[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 1, (cta.d1 * 2) * 256:(cta.d1 * 2) * 256 + 1]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                value_7 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_9 = T.tensor_view(
                    T.ptr_of(value_7),
                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                T.copy_async_tensor(value_8, value_9)
            value_10_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(106496, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            value_13_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            for kb in range(0, ((-2 * cta.d0) + (((32 * cta.d0) - 17) // 80)) + 14, 1):
                with scope[2:, :32] as scope_3:
                    value_11 = T.tensor_view(
                        T.ptr_of(qkv[kb * 80:kb * 80 + 1, 12288 + ((cta.d1 // 6) * 256):12288 + ((cta.d1 // 6) * 256) + 1]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_10_stages[kb % 2]),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    T.copy_async_tensor(value_11, value_12)
                    value_14 = T.tensor_view(
                        T.ptr_of(qkv[kb * 80:kb * 80 + 1, 13312 + ((cta.d1 // 6) * 256):13312 + ((cta.d1 // 6) * 256) + 1]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_13_stages[kb % 2]),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    T.copy_async_tensor(value_14, value_15)
                with Mesh(
                    (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    value_16 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    value_17 = T.tensor_view(
                        T.ptr_of(mx),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_17, scalar_2, value_16, kind=BinaryKind.MUL)
                    value_18 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    s0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(s0, 0.0)
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 80, 80):
                            for o_k in range(0, 256, 64):
                                value_7_2 = T.tensor_view(
                                    0,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                value_19 = T.tensor_view(
                                    T.ptr_of(value_7_2),
                                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 256),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_19[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_20 = T.tensor_view(
                                    T.ptr_of(value_10_stages[kb % 2]),
                                    layout=Layout(((4, 64), 80), ((5120, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(256, 80),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_20[o_k:o_k + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    s0,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_19[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_20[o_k + 16:o_k + 16 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    s0,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_19[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_20[o_k + 32:o_k + 32 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    s0,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_19[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_20[o_k + 48:o_k + 48 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    s0,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    for _dg in range(((-2 * cta.d0) + (((2 * cta.d0) + 1) // 5)) + 10, kb - (15 * (kb // 15)), 1048576):
                        value_21 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "i32",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_22 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (1, 80),
                                "i32",
                                ((1, 2, 4 @ scope_4.d3, 10), (0, 64, 128, 512)),
                                "rmem",
                            ]
                        )
                        value_23 = T.tensor_view(
                            T.ptr_of(iota), layout=((1, 80), (0, 1)), shape=(1, 80)
                        )
                        value_24 = T.tensor_view(
                            T.ptr_of(value_22),
                            layout=((1, 2, 4 @ scope_4.d3, 10), (0, 64, 128, 512)),
                            shape=(1, 80),
                        )
                        T.copy(value_23, value_24)
                        value_25 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"]
                        )
                        T.fill(value_25, 0.0)
                        value_26 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"]
                        )
                        scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                        T.fill(scalar_3, kb * 80)
                        T.binary(value_25, scalar_3, value_26, kind=BinaryKind.ADD)
                        value_27 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"]
                        )
                        T.cast(value_26, value_27, dtype="i32")
                        cols = T.alloc_tensor(
                            tensor_type=Tensor[
                                (1, 80),
                                "i32",
                                ((1, 2, 4 @ scope_4.d3, 10), (0, 64, 128, 512)),
                                "rmem",
                            ]
                        )
                        T.binary(value_24, value_27, cols, kind=BinaryKind.ADD)
                        value_28 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "i32",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 8 @ scope_4.d2, 2, 4 @ scope_4.d3, 20), (5120, 1280, 160, 80, 20, 1)),
                                "rmem",
                            ]
                        )
                        value_29 = T.tensor_view(
                            T.ptr_of(rows),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            shape=(128, 1),
                        )
                        T.binary(value_29, cols, value_28, kind=BinaryKind.SUB)
                        value_30 = T.tensor_view(
                            T.ptr_of(value_21),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        T.copy(value_28, value_30)
                        value_31 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"]
                        )
                        T.fill(value_31, 0.0)
                        value_32 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "bool",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        T.binary(value_30, value_31, value_32, kind=BinaryKind.GE)
                        scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_4, -1e+30)
                        value_33 = T.tensor_view(
                            T.ptr_of(s0),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        value_34 = T.tensor_view(
                            T.ptr_of(s0),
                            layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        T.where(value_32, value_33, scalar_4, value_34)
                    value_35 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1, 1, 1), (64, 16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_36 = T.tensor_view(
                        T.ptr_of(s0),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    T.reduce(value_36, value_35, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                    value_37 = T.tensor_view(
                        T.ptr_of(value_18),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.copy(value_35, value_37)
                    value_38 = T.tensor_view(
                        T.ptr_of(mx),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_17, value_37, value_38, kind=BinaryKind.MAX)
                    value_39 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_5, 0.09016844005556021)
                    T.binary(value_38, scalar_5, value_39, kind=BinaryKind.MUL)
                    value_40 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_16, value_39, value_40, kind=BinaryKind.SUB)
                    corr = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.unary(value_40, corr, kind=UnaryKind.EXP2)
                    value_41 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_42 = T.tensor_view(
                        T.ptr_of(lsum),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_42, corr, value_41, kind=BinaryKind.MUL)
                    value_43 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_44 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_6, 0.09016844005556021)
                    T.binary(value_36, scalar_6, value_44, kind=BinaryKind.MUL)
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_7, 0.09016844005556021)
                    T.binary(value_38, scalar_7, value_45, kind=BinaryKind.MUL)
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_44, value_45, value_46, kind=BinaryKind.SUB)
                    p = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_46, p, kind=UnaryKind.EXP2)
                    value_47 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1, 1, 1), (64, 16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(p, value_47, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    value_48 = T.tensor_view(
                        T.ptr_of(value_43),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.copy(value_47, value_48)
                    value_49 = T.tensor_view(
                        T.ptr_of(lsum),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_41, value_48, value_49, kind=BinaryKind.ADD)
                    value_50 = T.tensor_view(
                        T.ptr_of(o),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_51 = T.tensor_view(
                        T.ptr_of(o),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    T.binary(value_50, corr, value_51, kind=BinaryKind.MUL)
                    value_52 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 5, 2, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(p, value_53, dtype="bf16")
                    value_54 = T.tensor_view(
                        T.ptr_of(value_52),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 5, 2, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    T.copy(value_53, value_54)
                    for o_m_1 in range(0, 128, 128):
                        for o_n_1 in range(0, 256, 256):
                            for o_k_1 in range(0, 80, 16):
                                value_55 = T.tensor_view(
                                    T.ptr_of(o),
                                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_56 = T.tensor_view(
                                    T.ptr_of(value_52),
                                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 5, 2, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(128, 80),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_56[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 2, 4 @ scope_4.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 16),
                                )
                                value_57 = T.tensor_view(
                                    T.ptr_of(value_13_stages[kb % 2]),
                                    layout=Layout(((5, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(80, 256),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_57[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_55,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
            with Mesh(
                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_5:
                value_59 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_60 = T.tensor_view(
                    T.ptr_of(o),
                    layout=((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_61 = T.tensor_view(
                    T.ptr_of(lsum),
                    layout=((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.binary(value_60, value_61, value_59, kind=BinaryKind.DIV)
                value_62 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_59, value_62, dtype="bf16")
                ob = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_62, ob, dtype="f32")
                value_63 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_64 = T.tensor_view(
                    T.ptr_of(qkv[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 1, ((cta.d1 * 2) * 256) + 256:((cta.d1 * 2) * 256) + 256 + 1]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                value_65 = T.tensor_view(
                    T.ptr_of(value_63),
                    layout=((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                T.copy(value_64, value_65)
                gt = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_65, gt, dtype="f32")
                value_66 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(gt, value_66, kind=UnaryKind.NEG)
                value_67 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_66, value_67, kind=UnaryKind.EXP)
                value_68 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 1.0)
                T.binary(value_67, scalar_8, value_68, kind=BinaryKind.ADD)
                value_69 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 1.0)
                T.binary(scalar_9, value_68, value_69, kind=BinaryKind.DIV)
                value_70 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_69, value_70, dtype="bf16")
                sg = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_70, sg, dtype="f32")
                value_71 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(ob, sg, value_71, kind=BinaryKind.MUL)
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.d0, 4 @ scope_5.d1, 2, 8 @ scope_5.d2, 32, 4 @ scope_5.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_71, y, dtype="bf16")
                value_58 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_72 = T.tensor_view(
                    T.ptr_of(value_58),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                T.copy(y, value_72)
            with scope[:1, :32] as scope_6:
                value_58_2 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_74 = T.tensor_view(
                    T.ptr_of(value_58_2),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_73_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
                )
                value_75 = T.tensor_view(
                    T.ptr_of(value_73_1), layout=(128, 256), shape=(128, 256)
                )
                T.copy_async_tensor(value_74, value_75)
                value_76 = T.tensor_view(T.ptr_of(out), layout=(1024, 6144), shape=(1024, 6144))
                window = T.tensor_view(
                    T.ptr_of(value_76[((8 - 1) - cta.d0) * 128:((8 - 1) - cta.d0) * 128 + 128, cta.d1 * 256:cta.d1 * 256 + 256]),
                    layout=((128, 256), (6144, 1)),
                    shape=(128, 256),
                )
                T.copy(value_75, window)
