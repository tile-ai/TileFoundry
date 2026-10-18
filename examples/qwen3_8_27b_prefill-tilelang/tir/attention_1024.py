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
    with Mesh((Topology("cta", 192),), layout=(8, 24), names=("qb", "h")) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(1024, 6144), shape=(1024, 6144))
            T.fill(value, 0.0)
            o = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(1), S(3), S(5)),
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
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
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
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
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
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
                        mesh=Mesh((Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_1:
                T.fill(o, 0.0)
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 1),
                        "f32",
                        ((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                T.fill(value_1, 0.0)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1e+30)
                lhs_view = T.tensor_view(
                    T.ptr_of(value_1[0:0 + 128, 0:0 + 1]),
                    layout=((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.binary(lhs_view, scalar, mx, kind=BinaryKind.SUB)
                value_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 1),
                        "i32",
                        ((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                        "rmem",
                    ]
                )
                value_3 = T.tensor_view(T.ptr_of(iota), layout=(128,), shape=(128,))
                value_4 = T.tensor_view(
                    T.ptr_of(value_3), layout=((128, 1), (1, 0)), shape=(128, 1)
                )
                value_5 = T.tensor_view(
                    T.ptr_of(value_2),
                    layout=((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 128, 0:0 + 1]),
                    layout=((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.copy(value_4, dst_view)
                value_6 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
                T.fill(value_6, 0.0)
                value_7 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"])
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar_1, ((8 - 1) - cta.qb) * 128)
                T.binary(value_6, scalar_1, value_7, kind=BinaryKind.ADD)
                value_8 = T.alloc_tensor(tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"])
                T.cast(value_7, value_8, dtype="i32")
                lhs_view_1 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 128, 0:0 + 1]),
                    layout=((2 @ scope_1.group, 4 @ scope_1.warp, 2, 8 @ scope_1.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.binary(lhs_view_1, value_8, rows, kind=BinaryKind.ADD)
                T.fill(lsum, 0.0)
            with scope[2:, :32] as scope_2:
                value_10 = T.tensor_view(
                    T.ptr_of(qkv), layout=(1024, 14336), shape=(1024, 14336)
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value_10[((8 - 1) - cta.qb) * 128:((8 - 1) - cta.qb) * 128 + 1, (cta.h * 2) * 256:(cta.h * 2) * 256 + 1]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                value_9 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_9),
                    layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                T.copy_async_tensor(value_11, value_12)
            value_13_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(106496, dtype="bf16", storage="smem", layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            value_17_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
                T.tensor_view(188416, dtype="bf16", storage="smem", layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(80, 256)),
            )
            for kb in range(0, ((-2 * cta.qb) + (((32 * cta.qb) - 17) // 80)) + 14, 1):
                with scope[2:, :32] as scope_3:
                    value_14 = T.tensor_view(
                        T.ptr_of(qkv), layout=(1024, 14336), shape=(1024, 14336)
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14[kb * 80:kb * 80 + 1, 12288 + ((cta.h // 6) * 256):12288 + ((cta.h // 6) * 256) + 1]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_13_stages[kb % 2]),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    T.copy_async_tensor(value_15, value_16)
                    value_18 = T.tensor_view(
                        T.ptr_of(value_14[kb * 80:kb * 80 + 1, 13312 + ((cta.h // 6) * 256):13312 + ((cta.h // 6) * 256) + 1]),
                        layout=((80, 256), (14336, 1)),
                        shape=(80, 256),
                    )
                    value_19 = T.tensor_view(
                        T.ptr_of(value_17_stages[kb % 2]),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    T.copy_async_tensor(value_18, value_19)
                with Mesh(
                    (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_4:
                    value_20 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    value_21 = T.tensor_view(
                        T.ptr_of(mx),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_22 = T.tensor_view(
                        T.ptr_of(value_21),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_23 = T.tensor_view(
                        T.ptr_of(value_22),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_24 = T.tensor_view(
                        T.ptr_of(value_23),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_25 = T.tensor_view(
                        T.ptr_of(value_24),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_25, scalar_2, value_20, kind=BinaryKind.MUL)
                    value_26 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    s0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(s0, 0.0)
                    acc_view = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    value_9_2 = T.tensor_view(
                        0,
                        dtype="bf16",
                        storage="smem",
                        layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    value_27 = T.tensor_view(
                        T.ptr_of(value_9_2),
                        layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    value_28 = T.tensor_view(
                        T.ptr_of(value_27),
                        layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    value_29 = T.tensor_view(
                        T.ptr_of(value_28),
                        layout=Layout(((2, 8, 8), (4, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 256),
                    )
                    lhs_view_2 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 0:0 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    value_30 = T.tensor_view(
                        T.ptr_of(value_13_stages[kb % 2]),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_31 = T.tensor_view(
                        T.ptr_of(value_30),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_32 = T.tensor_view(
                        T.ptr_of(value_31),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_33 = T.tensor_view(
                        T.ptr_of(value_32),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_34 = T.tensor_view(
                        T.ptr_of(value_33),
                        layout=Layout((80, (4, 64)), (64, (5120, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_35 = T.tensor_view(
                        T.ptr_of(value_34),
                        layout=Layout(((4, 64), 80), ((5120, 1), 64)) | Swizzle(3, 4, 3),
                        shape=(256, 80),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view_2,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_3 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 16:16 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_35[16:16 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        lhs_view_3,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_2 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_4 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 32:32 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_2 = T.tensor_view(
                        T.ptr_of(value_35[32:32 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_2,
                        lhs_view_4,
                        rhs_view_2,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_3 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_5 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 48:48 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_3 = T.tensor_view(
                        T.ptr_of(value_35[48:48 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_3,
                        lhs_view_5,
                        rhs_view_3,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_4 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_6 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 64:64 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_4 = T.tensor_view(
                        T.ptr_of(value_35[64:64 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_4,
                        lhs_view_6,
                        rhs_view_4,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_5 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_7 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 80:80 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_5 = T.tensor_view(
                        T.ptr_of(value_35[80:80 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_5,
                        lhs_view_7,
                        rhs_view_5,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_6 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_8 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 96:96 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_6 = T.tensor_view(
                        T.ptr_of(value_35[96:96 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_6,
                        lhs_view_8,
                        rhs_view_6,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_7 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_9 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 112:112 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_7 = T.tensor_view(
                        T.ptr_of(value_35[112:112 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_7,
                        lhs_view_9,
                        rhs_view_7,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_8 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_10 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 128:128 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_8 = T.tensor_view(
                        T.ptr_of(value_35[128:128 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_8,
                        lhs_view_10,
                        rhs_view_8,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_9 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_11 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 144:144 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_9 = T.tensor_view(
                        T.ptr_of(value_35[144:144 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_9,
                        lhs_view_11,
                        rhs_view_9,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_10 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_12 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 160:160 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_10 = T.tensor_view(
                        T.ptr_of(value_35[160:160 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_10,
                        lhs_view_12,
                        rhs_view_10,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_11 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_13 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 176:176 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_11 = T.tensor_view(
                        T.ptr_of(value_35[176:176 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_11,
                        lhs_view_13,
                        rhs_view_11,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_12 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_14 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 192:192 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_12 = T.tensor_view(
                        T.ptr_of(value_35[192:192 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_12,
                        lhs_view_14,
                        rhs_view_12,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_13 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_15 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 208:208 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_13 = T.tensor_view(
                        T.ptr_of(value_35[208:208 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_13,
                        lhs_view_15,
                        rhs_view_13,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_14 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_16 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 224:224 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_14 = T.tensor_view(
                        T.ptr_of(value_35[224:224 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_14,
                        lhs_view_16,
                        rhs_view_14,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    acc_view_15 = T.tensor_view(
                        T.ptr_of(s0[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    lhs_view_17 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 128, 240:240 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_15 = T.tensor_view(
                        T.ptr_of(value_35[240:240 + 16, 0:0 + 80]),
                        layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 80),
                    )
                    T.tiled_mma(
                        acc_view_15,
                        lhs_view_17,
                        rhs_view_15,
                        atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
                    for _dg in range(((-2 * cta.qb) + (((2 * cta.qb) + 1) // 5)) + 10, kb - (15 * (kb // 15)), 1048576):
                        value_36 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "i32",
                                ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        value_37 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (1, 80),
                                "i32",
                                ((1, 2, 4 @ scope_4.lane4, 10), (0, 64, 128, 512)),
                                "rmem",
                            ]
                        )
                        value_38 = T.tensor_view(T.ptr_of(iota), layout=(128,), shape=(128,))
                        value_39 = T.tensor_view(T.ptr_of(value_38), layout=(80,), shape=(80,))
                        value_40 = T.tensor_view(
                            T.ptr_of(value_39), layout=((1, 80), (0, 1)), shape=(1, 80)
                        )
                        value_41 = T.tensor_view(
                            T.ptr_of(value_37),
                            layout=((1, 2, 4 @ scope_4.lane4, 10), (0, 64, 128, 512)),
                            shape=(1, 80),
                        )
                        dst_view_1 = T.tensor_view(
                            T.ptr_of(value_41[0:0 + 1, 0:0 + 80]),
                            layout=((1, 2, 4 @ scope_4.lane4, 10), (0, 64, 128, 512)),
                            shape=(1, 80),
                        )
                        T.copy(value_40, dst_view_1)
                        value_42 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"]
                        )
                        T.fill(value_42, 0.0)
                        value_43 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i64", (1, 1), "rmem"]
                        )
                        scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                        T.fill(scalar_3, kb * 80)
                        T.binary(value_42, scalar_3, value_43, kind=BinaryKind.ADD)
                        value_44 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"]
                        )
                        T.cast(value_43, value_44, dtype="i32")
                        cols = T.alloc_tensor(
                            tensor_type=Tensor[
                                (1, 80),
                                "i32",
                                ((1, 2, 4 @ scope_4.lane4, 10), (0, 64, 128, 512)),
                                "rmem",
                            ]
                        )
                        lhs_view_18 = T.tensor_view(
                            T.ptr_of(value_41[0:0 + 1, 0:0 + 80]),
                            layout=((1, 2, 4 @ scope_4.lane4, 10), (0, 64, 128, 512)),
                            shape=(1, 80),
                        )
                        T.binary(lhs_view_18, value_44, cols, kind=BinaryKind.ADD)
                        value_45 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "i32",
                                ((2 @ scope_4.group, 4 @ scope_4.warp, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 20), (5120, 1280, 160, 80, 20, 1)),
                                "rmem",
                            ]
                        )
                        value_46 = T.tensor_view(
                            T.ptr_of(rows),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            shape=(128, 1),
                        )
                        value_47 = T.tensor_view(
                            T.ptr_of(value_46),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            shape=(128, 1),
                        )
                        value_48 = T.tensor_view(
                            T.ptr_of(value_47),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            shape=(128, 1),
                        )
                        value_49 = T.tensor_view(
                            T.ptr_of(value_48),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            shape=(128, 1),
                        )
                        T.binary(value_49, cols, value_45, kind=BinaryKind.SUB)
                        value_50 = T.tensor_view(
                            T.ptr_of(value_36),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        dst_view_2 = T.tensor_view(
                            T.ptr_of(value_50[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        T.copy(value_45, dst_view_2)
                        value_51 = T.alloc_tensor(
                            tensor_type=Tensor[(1, 1), "i32", (1, 1), "rmem"]
                        )
                        T.fill(value_51, 0.0)
                        value_52 = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 80),
                                "bool",
                                ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        lhs_view_19 = T.tensor_view(
                            T.ptr_of(value_50[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        T.binary(lhs_view_19, value_51, value_52, kind=BinaryKind.GE)
                        scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                        T.fill(scalar_4, -1e+30)
                        value_53 = T.tensor_view(
                            T.ptr_of(s0),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        value_54 = T.tensor_view(
                            T.ptr_of(value_53),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        lhs_view_20 = T.tensor_view(
                            T.ptr_of(value_54[0:0 + 128, 0:0 + 80]),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        value_55 = T.tensor_view(
                            T.ptr_of(value_54),
                            layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            shape=(128, 80),
                        )
                        T.where(value_52, lhs_view_20, scalar_4, value_55)
                    value_56 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1, 1, 1), (64, 16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_57 = T.tensor_view(
                        T.ptr_of(s0),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    value_58 = T.tensor_view(
                        T.ptr_of(value_57),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    value_59 = T.tensor_view(
                        T.ptr_of(value_58),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    src_view = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    T.reduce(src_view, value_56, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                    value_60 = T.tensor_view(
                        T.ptr_of(value_26),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    dst_view_3 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.copy(value_56, dst_view_3)
                    rhs_view_16 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_61 = T.tensor_view(
                        T.ptr_of(value_24),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_25, rhs_view_16, value_61, kind=BinaryKind.MAX)
                    value_62 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_5, 0.09016844005556021)
                    T.binary(value_61, scalar_5, value_62, kind=BinaryKind.MUL)
                    value_63 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_20, value_62, value_63, kind=BinaryKind.SUB)
                    corr = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.unary(value_63, corr, kind=UnaryKind.EXP2)
                    value_64 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(lsum),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_66 = T.tensor_view(
                        T.ptr_of(value_65),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_67 = T.tensor_view(
                        T.ptr_of(value_66),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_68 = T.tensor_view(
                        T.ptr_of(value_67),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_69 = T.tensor_view(
                        T.ptr_of(value_68),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    lhs_view_21 = T.tensor_view(
                        T.ptr_of(value_69[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(lhs_view_21, corr, value_64, kind=BinaryKind.MUL)
                    value_70 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    value_71 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_6, 0.09016844005556021)
                    lhs_view_22 = T.tensor_view(
                        T.ptr_of(value_59[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    T.binary(lhs_view_22, scalar_6, value_71, kind=BinaryKind.MUL)
                    value_72 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                            "rmem",
                        ]
                    )
                    scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_7, 0.09016844005556021)
                    T.binary(value_61, scalar_7, value_72, kind=BinaryKind.MUL)
                    value_73 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_71, value_72, value_73, kind=BinaryKind.SUB)
                    p = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_73, p, kind=UnaryKind.EXP2)
                    value_74 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 1),
                            "f32",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1, 1, 1), (64, 16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(p, value_74, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    value_75 = T.tensor_view(
                        T.ptr_of(value_70),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    dst_view_4 = T.tensor_view(
                        T.ptr_of(value_75[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.copy(value_74, dst_view_4)
                    rhs_view_17 = T.tensor_view(
                        T.ptr_of(value_75[0:0 + 128, 0:0 + 1]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    value_76 = T.tensor_view(
                        T.ptr_of(value_68),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 1), (64, 16, 8, 1, 0)),
                        shape=(128, 1),
                    )
                    T.binary(value_64, rhs_view_17, value_76, kind=BinaryKind.ADD)
                    value_77 = T.tensor_view(
                        T.ptr_of(o),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_78 = T.tensor_view(
                        T.ptr_of(value_77),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_79 = T.tensor_view(
                        T.ptr_of(value_78),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_80 = T.tensor_view(
                        T.ptr_of(value_79),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_81 = T.tensor_view(
                        T.ptr_of(value_80),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    lhs_view_23 = T.tensor_view(
                        T.ptr_of(value_81[0:0 + 128, 0:0 + 256]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_82 = T.tensor_view(
                        T.ptr_of(value_80),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 32, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    T.binary(lhs_view_23, corr, value_82, kind=BinaryKind.MUL)
                    value_83 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "bf16",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 5, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_84 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 80),
                            "bf16",
                            ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 10, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(p, value_84, dtype="bf16")
                    value_85 = T.tensor_view(
                        T.ptr_of(value_83),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 5, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    dst_view_5 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 0:0 + 80]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 5, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 1024, 512, 128, 64)),
                        shape=(128, 80),
                    )
                    T.copy(value_84, dst_view_5)
                    lhs_view_24 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    value_86 = T.tensor_view(
                        T.ptr_of(value_17_stages[kb % 2]),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_87 = T.tensor_view(
                        T.ptr_of(value_86),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_88 = T.tensor_view(
                        T.ptr_of(value_87),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_89 = T.tensor_view(
                        T.ptr_of(value_88),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_90 = T.tensor_view(
                        T.ptr_of(value_89),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    value_91 = T.tensor_view(
                        T.ptr_of(value_90),
                        layout=Layout(((5, 16), (4, 64)), ((4096, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 256),
                    )
                    rhs_view_18 = T.tensor_view(
                        T.ptr_of(value_91[0:0 + 16, 0:0 + 256]),
                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 256),
                    )
                    T.tiled_mma(
                        value_82,
                        lhs_view_24,
                        rhs_view_18,
                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_25 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 16:16 + 16]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    rhs_view_19 = T.tensor_view(
                        T.ptr_of(value_91[16:16 + 16, 0:0 + 256]),
                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 256),
                    )
                    T.tiled_mma(
                        value_82,
                        lhs_view_25,
                        rhs_view_19,
                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_26 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 32:32 + 16]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    rhs_view_20 = T.tensor_view(
                        T.ptr_of(value_91[32:32 + 16, 0:0 + 256]),
                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 256),
                    )
                    T.tiled_mma(
                        value_82,
                        lhs_view_26,
                        rhs_view_20,
                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_27 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 48:48 + 16]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    rhs_view_21 = T.tensor_view(
                        T.ptr_of(value_91[48:48 + 16, 0:0 + 256]),
                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 256),
                    )
                    T.tiled_mma(
                        value_82,
                        lhs_view_27,
                        rhs_view_21,
                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    lhs_view_28 = T.tensor_view(
                        T.ptr_of(value_85[0:0 + 128, 64:64 + 16]),
                        layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (5120, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    rhs_view_22 = T.tensor_view(
                        T.ptr_of(value_91[64:64 + 16, 0:0 + 256]),
                        layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(16, 256),
                    )
                    T.tiled_mma(
                        value_82,
                        lhs_view_28,
                        rhs_view_22,
                        atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            with Mesh(
                (Topology("thread", 384),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_5:
                value_93 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_94 = T.tensor_view(
                    T.ptr_of(o),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_95 = T.tensor_view(
                    T.ptr_of(value_94),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_96 = T.tensor_view(
                    T.ptr_of(value_95),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_97 = T.tensor_view(
                    T.ptr_of(value_96),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_98 = T.tensor_view(
                    T.ptr_of(value_97),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_99 = T.tensor_view(
                    T.ptr_of(value_98),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_100 = T.tensor_view(
                    T.ptr_of(value_99),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                lhs_view_29 = T.tensor_view(
                    T.ptr_of(value_100[0:0 + 128, 0:0 + 256]),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_101 = T.tensor_view(
                    T.ptr_of(lsum),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_102 = T.tensor_view(
                    T.ptr_of(value_101),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_103 = T.tensor_view(
                    T.ptr_of(value_102),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_104 = T.tensor_view(
                    T.ptr_of(value_103),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_105 = T.tensor_view(
                    T.ptr_of(value_104),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_106 = T.tensor_view(
                    T.ptr_of(value_105),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                value_107 = T.tensor_view(
                    T.ptr_of(value_106),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                rhs_view_23 = T.tensor_view(
                    T.ptr_of(value_107[0:0 + 128, 0:0 + 1]),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1), (64, 16, 8, 1, 0)),
                    shape=(128, 1),
                )
                T.binary(lhs_view_29, rhs_view_23, value_93, kind=BinaryKind.DIV)
                value_108 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_93, value_108, dtype="bf16")
                ob = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_108, ob, dtype="f32")
                value_109 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_110 = T.tensor_view(
                    T.ptr_of(qkv), layout=(1024, 14336), shape=(1024, 14336)
                )
                value_111 = T.tensor_view(
                    T.ptr_of(value_110[((8 - 1) - cta.qb) * 128:((8 - 1) - cta.qb) * 128 + 1, ((cta.h * 2) * 256) + 256:((cta.h * 2) * 256) + 256 + 1]),
                    layout=((128, 256), (14336, 1)),
                    shape=(128, 256),
                )
                value_112 = T.tensor_view(
                    T.ptr_of(value_109),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                dst_view_6 = T.tensor_view(
                    T.ptr_of(value_112[0:0 + 128, 0:0 + 256]),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                T.copy(value_111, dst_view_6)
                gt = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_112[0:0 + 128, 0:0 + 256]),
                    layout=((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                T.cast(src_view_1, gt, dtype="f32")
                value_113 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(gt, value_113, kind=UnaryKind.NEG)
                value_114 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_113, value_114, kind=UnaryKind.EXP)
                value_115 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 1.0)
                T.binary(value_114, scalar_8, value_115, kind=BinaryKind.ADD)
                value_116 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 1.0)
                T.binary(scalar_9, value_115, value_116, kind=BinaryKind.DIV)
                value_117 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_116, value_117, dtype="bf16")
                sg = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_117, sg, dtype="f32")
                value_118 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "f32",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(ob, sg, value_118, kind=BinaryKind.MUL)
                y = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_5.group, 4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 32, 4 @ scope_5.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_118, y, dtype="bf16")
                value_92 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_119 = T.tensor_view(
                    T.ptr_of(value_92),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                T.copy(y, value_119)
            with scope[:1, :32] as scope_6:
                value_92_2 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_121 = T.tensor_view(
                    T.ptr_of(value_92_2),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_122 = T.tensor_view(
                    T.ptr_of(value_121),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_123 = T.tensor_view(
                    T.ptr_of(value_122),
                    layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                    shape=(128, 256),
                )
                value_120_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
                )
                value_124 = T.tensor_view(
                    T.ptr_of(value_120_1), layout=(128, 256), shape=(128, 256)
                )
                T.copy_async_tensor(value_123, value_124)
                value_125 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 6144), shape=(1024, 6144)
                )
                window = T.tensor_view(
                    T.ptr_of(value_125[((8 - 1) - cta.qb) * 128:((8 - 1) - cta.qb) * 128 + 128, cta.h * 256:cta.h * 256 + 256]),
                    layout=((128, 256), (6144, 1)),
                    shape=(128, 256),
                )
                T.copy(value_124, window)
