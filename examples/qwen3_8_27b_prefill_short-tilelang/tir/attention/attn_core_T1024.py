# analysis target=nvidia.h200_sxm module=SCHED_ATTN_CORE_T1024 function=attn_core topology=cta wave=132/192
# selection requested=memory executed=memory
# memory traffic=gmem:r1.22MB/w12.12MB@logical,r234.00MB/w2.27GB@total,r1.22MB/w12.12MB@cta,r1.19MB/w12.12MB@thread;rmem:r9.73MB/w8.23MB@logical,r1.82GB/w1.62GB@total,r9.73MB/w8.67MB@cta,r81.74KB/w71.14KB@thread;smem:r3.12MB/w1.19MB@logical,r600.00MB/w228.00MB@total,r3.12MB/w1.19MB@cta,r416.50KB/w1.13MB@thread footprint=diag:16.00KB;k:1.50MB;q:8.25MB;qkv:8.25MB;v:1.50MB;v216:141:32.00KB;v222:142:4.12MB;v223:143:32.00KB;v229:144:4.12MB footprint-precision=exact peak=gmem:56.05MB;rmem:240.52KB;smem:224.00KB persistent=gmem:44.02MB
#   buffer=k holds=27.83MB time=none space=cta.blk reuse=15.00MB fits=yes precision=exact
#   buffer=v holds=27.83MB time=none space=cta.blk reuse=15.00MB fits=yes precision=exact
#   buffer=diag holds=27.83MB time=none space=cta.blk,cta.hg reuse=2.05MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_core(
    q: Tensor[(1024, 24, 256), "bf16"], k: Tensor[(1024, 4, 256), "bf16"], v: Tensor[(1024, 4, 256), "bf16"], qkv: Tensor[(1024, 14336), "bf16"], diag: Tensor[(64, 64), "f32"], scratch: Tensor[(16384,), "bf16"], scratch_1: Tensor[(16384,), "bf16"], out: Tensor[(1024, 6144), "bf16"]
):
    k_ring_1_stages = (
        T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
    )
    v_ring_1_stages = (
        T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
    )
    with Mesh((Topology("cta", 192),), layout=(16, 12), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            O0 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            m0 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            l0 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 8, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_1:
                T.fill(O0, 0.0)
                S0_0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_1.d0, 2, 8 @ scope_1.d1, 8, 4 @ scope_1.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(S0_0, 0.0)
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_1.d0, 2, 8 @ scope_1.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(S0_0, value, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, -1e999)
                T.binary(value, scalar, m0, kind=BinaryKind.ADD)
                T.reduce(S0_0, l0, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                value_2 = T.tensor_view(T.ptr_of(diag), layout=(64, 64), shape=(64, 64))
                value_3 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((4 @ scope_1.d0, 2, 8 @ scope_1.d1, 8, 4 @ scope_1.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_2, value_3)
            O1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            m1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            l1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 64),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 8, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_2:
                T.fill(O1, 0.0)
                S0_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_2.d0, 2, 8 @ scope_2.d1, 8, 4 @ scope_2.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(S0_1, 0.0)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_2.d0, 2, 8 @ scope_2.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(S0_1, value_4, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, -1e999)
                T.binary(value_4, scalar_1, m1, kind=BinaryKind.ADD)
                T.reduce(S0_1, l1, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                value_6 = T.tensor_view(T.ptr_of(diag), layout=(64, 64), shape=(64, 64))
                value_7 = T.tensor_view(
                    T.ptr_of(value_5),
                    layout=((4 @ scope_2.d0, 2, 8 @ scope_2.d1, 8, 4 @ scope_2.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_6, value_7)
            with scope[:1, :32] as scope_3:
                value_8 = T.tensor_view(
                    T.ptr_of(q[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, (cta.d1 * 2) + 0:(cta.d1 * 2) + 0 + 1, 0:0 + 1]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                q_dst0 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_9 = T.tensor_view(
                    T.ptr_of(q_dst0),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_8, value_9)
                value_10 = T.tensor_view(
                    T.ptr_of(q[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, (cta.d1 * 2) + 1:(cta.d1 * 2) + 1 + 1, 0:0 + 1]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                q_dst1 = T.tensor_view(
                    32768,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_11 = T.tensor_view(
                    T.ptr_of(q_dst1),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_10, value_11)
            k_ring_stages = (
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            v_ring_stages = (
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for j in range(0, (0 - cta.d0) + 15, 1):
                with scope[:1, :32] as scope_4:
                    value_12 = T.tensor_view(
                        T.ptr_of(k[j * 64:j * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(k_ring_stages[j % 2]),
                        layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 256),
                    )
                    T.copy_async_tensor(value_12, value_13)
                with scope[:1, 32:64] as scope_5:
                    value_14 = T.tensor_view(
                        T.ptr_of(v[j * 64:j * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(v_ring_stages[j % 2]),
                        layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 256),
                    )
                    T.copy_async_tensor(value_14, value_15)
                m_new1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ShardLayout(
                            layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            attrs=(S(0), S(2), B()),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=256,
                                outer=Layout((4, 8, 4), (32, 4, 1)),
                            ), names=("d0", "d1", "d2")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_6:
                    value_16 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(value_16, 0.0)
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 64, 64):
                            for o_k in range(0, 256, 64):
                                q_dst1_2 = T.tensor_view(
                                    32768,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                value_17 = T.tensor_view(
                                    T.ptr_of(q_dst1_2),
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_17[o_m:o_m + 64, o_k:o_k + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_18 = T.tensor_view(
                                    T.ptr_of(k_ring_stages[j % 2]),
                                    layout=Layout(((4, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_18[o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_16,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_17[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_18[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_16,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_17[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_18[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_16,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_17[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_18[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_16,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_20 = T.tensor_view(
                        T.ptr_of(value_16),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.reduce(value_20, value_19, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                    value_21 = T.tensor_view(
                        T.ptr_of(m1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_21, value_19, m_new1, kind=BinaryKind.MAX)
                    value_22 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_21, m_new1, value_22, kind=BinaryKind.SUB)
                    value_23 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    T.binary(value_22, scalar_2, value_23, kind=BinaryKind.MUL)
                    alpha1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.unary(value_23, alpha1, kind=UnaryKind.EXP2)
                    value_24 = T.tensor_view(
                        T.ptr_of(O1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 32, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    value_25 = T.tensor_view(
                        T.ptr_of(O1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 32, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    T.binary(value_24, alpha1, value_25, kind=BinaryKind.MUL)
                    value_26 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 4, 2, 4 @ scope_6.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_27 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_3, 0.09016844005556021)
                    T.binary(value_20, scalar_3, value_27, kind=BinaryKind.MUL)
                    value_28 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_4, 0.09016844005556021)
                    T.binary(m_new1, scalar_4, value_28, kind=BinaryKind.MUL)
                    value_29 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_27, value_28, value_29, kind=BinaryKind.SUB)
                    P1 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_29, P1, kind=UnaryKind.EXP2)
                    value_30 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 8, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(P1, value_30, dtype="bf16")
                    value_31 = T.tensor_view(
                        T.ptr_of(value_26),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 4, 2, 4 @ scope_6.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_30, value_31)
                    for o_m_1 in range(0, 64, 64):
                        for o_n_1 in range(0, 256, 256):
                            for o_k_1 in range(0, 64, 16):
                                value_32 = T.tensor_view(
                                    T.ptr_of(O1),
                                    layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 32, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 256),
                                )
                                value_33 = T.tensor_view(
                                    T.ptr_of(value_26),
                                    layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 4, 2, 4 @ scope_6.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_33[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                    layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 2, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_34 = T.tensor_view(
                                    T.ptr_of(v_ring_stages[j % 2]),
                                    layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_34[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_32,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_35 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_36 = T.tensor_view(
                        T.ptr_of(l1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_36, alpha1, value_35, kind=BinaryKind.MUL)
                    value_37 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(P1, value_37, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                    value_38 = T.tensor_view(
                        T.ptr_of(l1),
                        layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_35, value_37, value_38, kind=BinaryKind.ADD)
                m_new0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ShardLayout(
                            layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            attrs=(S(0), S(2), B()),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((4, 8, 4), (32, 4, 1)),
                            ), names=("d0", "d1", "d2")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_7:
                    value_39 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(value_39, 0.0)
                    for o_m_2 in range(0, 64, 64):
                        for o_n_2 in range(0, 64, 64):
                            for o_k_2 in range(0, 256, 64):
                                q_dst0_2 = T.tensor_view(
                                    0,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                value_40 = T.tensor_view(
                                    T.ptr_of(q_dst0_2),
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                lhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_40[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_41 = T.tensor_view(
                                    T.ptr_of(k_ring_stages[j % 2]),
                                    layout=Layout(((4, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                rhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_41[o_k_2:o_k_2 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_39,
                                    lhs_view_5,
                                    rhs_view_5,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_40[o_m_2:o_m_2 + 64, o_k_2 + 16:o_k_2 + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_41[o_k_2 + 16:o_k_2 + 16 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_39,
                                    lhs_view_6,
                                    rhs_view_6,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_40[o_m_2:o_m_2 + 64, o_k_2 + 32:o_k_2 + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_41[o_k_2 + 32:o_k_2 + 32 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_39,
                                    lhs_view_7,
                                    rhs_view_7,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_40[o_m_2:o_m_2 + 64, o_k_2 + 48:o_k_2 + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_8 = T.tensor_view(
                                    T.ptr_of(value_41[o_k_2 + 48:o_k_2 + 48 + 16, o_n_2:o_n_2 + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_39,
                                    lhs_view_8,
                                    rhs_view_8,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_42 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_43 = T.tensor_view(
                        T.ptr_of(value_39),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.reduce(value_43, value_42, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                    value_44 = T.tensor_view(
                        T.ptr_of(m0),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_44, value_42, m_new0, kind=BinaryKind.MAX)
                    value_45 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_44, m_new0, value_45, kind=BinaryKind.SUB)
                    value_46 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_5, 0.09016844005556021)
                    T.binary(value_45, scalar_5, value_46, kind=BinaryKind.MUL)
                    alpha0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.unary(value_46, alpha0, kind=UnaryKind.EXP2)
                    value_47 = T.tensor_view(
                        T.ptr_of(O0),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 32, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    value_48 = T.tensor_view(
                        T.ptr_of(O0),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 32, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    T.binary(value_47, alpha0, value_48, kind=BinaryKind.MUL)
                    value_49 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 4, 2, 4 @ scope_7.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_50 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_6, 0.09016844005556021)
                    T.binary(value_43, scalar_6, value_50, kind=BinaryKind.MUL)
                    value_51 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_7, 0.09016844005556021)
                    T.binary(m_new0, scalar_7, value_51, kind=BinaryKind.MUL)
                    value_52 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_50, value_51, value_52, kind=BinaryKind.SUB)
                    P0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_52, P0, kind=UnaryKind.EXP2)
                    value_53 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 8, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(P0, value_53, dtype="bf16")
                    value_54 = T.tensor_view(
                        T.ptr_of(value_49),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 4, 2, 4 @ scope_7.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_53, value_54)
                    for o_m_3 in range(0, 64, 64):
                        for o_n_3 in range(0, 256, 256):
                            for o_k_3 in range(0, 64, 16):
                                value_55 = T.tensor_view(
                                    T.ptr_of(O0),
                                    layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 32, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 256),
                                )
                                value_56 = T.tensor_view(
                                    T.ptr_of(value_49),
                                    layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 4, 2, 4 @ scope_7.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_56[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                    layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 2, 4 @ scope_7.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_57 = T.tensor_view(
                                    T.ptr_of(v_ring_stages[j % 2]),
                                    layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view_9 = T.tensor_view(
                                    T.ptr_of(value_57[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_55,
                                    lhs_view_9,
                                    rhs_view_9,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_58 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_59 = T.tensor_view(
                        T.ptr_of(l0),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_59, alpha0, value_58, kind=BinaryKind.MUL)
                    value_60 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(P0, value_60, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                    value_61 = T.tensor_view(
                        T.ptr_of(l0),
                        layout=((4 @ scope_7.d0, 2, 8 @ scope_7.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_58, value_60, value_61, kind=BinaryKind.ADD)
                with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                    value_62 = T.tensor_view(
                        T.ptr_of(m_new0),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    value_63 = T.tensor_view(
                        T.ptr_of(m0),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_62, value_63)
                with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads_1:
                    value_64 = T.tensor_view(
                        T.ptr_of(m_new1),
                        layout=((4 @ threads_1.d0, 2, 8 @ threads_1.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    value_65 = T.tensor_view(
                        T.ptr_of(m1),
                        layout=((4 @ threads_1.d0, 2, 8 @ threads_1.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_64, value_65)
            with scope[:1, :32] as scope_8:
                value_66 = T.tensor_view(
                    T.ptr_of(k[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                value_67 = T.tensor_view(
                    T.ptr_of(k_ring_1_stages[0]),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_66, value_67)
            with scope[:1, 32:64] as scope_9:
                value_68 = T.tensor_view(
                    T.ptr_of(v[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, cta.d1 // 3:cta.d1 // 3 + 1, 0:0 + 1]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                value_69 = T.tensor_view(
                    T.ptr_of(v_ring_1_stages[0]),
                    layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_68, value_69)
            m_new0_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_79 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            l0_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_10:
                value_70 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_70, 0.0)
                for o_m_4 in range(0, 64, 64):
                    for o_n_4 in range(0, 64, 64):
                        for o_k_4 in range(0, 256, 64):
                            q_dst0_3 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            value_71 = T.tensor_view(
                                T.ptr_of(q_dst0_3),
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            lhs_view_10 = T.tensor_view(
                                T.ptr_of(value_71[o_m_4:o_m_4 + 64, o_k_4:o_k_4 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_72 = T.tensor_view(
                                T.ptr_of(k_ring_1_stages[0]),
                                layout=Layout(((4, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            rhs_view_10 = T.tensor_view(
                                T.ptr_of(value_72[o_k_4:o_k_4 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_70,
                                lhs_view_10,
                                rhs_view_10,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_11 = T.tensor_view(
                                T.ptr_of(value_71[o_m_4:o_m_4 + 64, o_k_4 + 16:o_k_4 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_11 = T.tensor_view(
                                T.ptr_of(value_72[o_k_4 + 16:o_k_4 + 16 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_70,
                                lhs_view_11,
                                rhs_view_11,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_12 = T.tensor_view(
                                T.ptr_of(value_71[o_m_4:o_m_4 + 64, o_k_4 + 32:o_k_4 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_12 = T.tensor_view(
                                T.ptr_of(value_72[o_k_4 + 32:o_k_4 + 32 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_70,
                                lhs_view_12,
                                rhs_view_12,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_13 = T.tensor_view(
                                T.ptr_of(value_71[o_m_4:o_m_4 + 64, o_k_4 + 48:o_k_4 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_13 = T.tensor_view(
                                T.ptr_of(value_72[o_k_4 + 48:o_k_4 + 48 + 16, o_n_4:o_n_4 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_70,
                                lhs_view_13,
                                rhs_view_13,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                S0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_73 = T.tensor_view(
                    T.ptr_of(value_70),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                value_74 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_73, value_74, S0, kind=BinaryKind.ADD)
                value_75 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(S0, value_75, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                value_76 = T.tensor_view(
                    T.ptr_of(m0),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_76, value_75, m_new0_1, kind=BinaryKind.MAX)
                value_77 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.binary(value_76, m_new0_1, value_77, kind=BinaryKind.SUB)
                value_78 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 0.09016844005556021)
                T.binary(value_77, scalar_8, value_78, kind=BinaryKind.MUL)
                alpha0_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.unary(value_78, alpha0_1, kind=UnaryKind.EXP2)
                value_80 = T.tensor_view(
                    T.ptr_of(O0),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 32, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_80, alpha0_1, value_79, kind=BinaryKind.MUL)
                value_81 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 4, 2, 4 @ scope_10.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_82 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 0.09016844005556021)
                T.binary(S0, scalar_9, value_82, kind=BinaryKind.MUL)
                value_83 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_10 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_10, 0.09016844005556021)
                T.binary(m_new0_1, scalar_10, value_83, kind=BinaryKind.MUL)
                value_84 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_82, value_83, value_84, kind=BinaryKind.SUB)
                P0_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_84, P0_1, kind=UnaryKind.EXP2)
                value_85 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 8, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(P0_1, value_85, dtype="bf16")
                value_86 = T.tensor_view(
                    T.ptr_of(value_81),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 4, 2, 4 @ scope_10.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_85, value_86)
                for o_m_5 in range(0, 64, 64):
                    for o_n_5 in range(0, 256, 256):
                        for o_k_5 in range(0, 64, 16):
                            value_87 = T.tensor_view(
                                T.ptr_of(value_81),
                                layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 4, 2, 4 @ scope_10.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_14 = T.tensor_view(
                                T.ptr_of(value_87[o_m_5:o_m_5 + 64, o_k_5:o_k_5 + 16]),
                                layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 2, 4 @ scope_10.d2, 2), (16, 8, 1, 512, 128, 64)),
                                shape=(64, 16),
                            )
                            value_88 = T.tensor_view(
                                T.ptr_of(v_ring_1_stages[0]),
                                layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            rhs_view_14 = T.tensor_view(
                                T.ptr_of(value_88[o_k_5:o_k_5 + 16, o_n_5:o_n_5 + 256]),
                                layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                value_79,
                                lhs_view_14,
                                rhs_view_14,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_89 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_90 = T.tensor_view(
                    T.ptr_of(l0),
                    layout=((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_90, alpha0_1, value_89, kind=BinaryKind.MUL)
                value_91 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_10.d0, 2, 8 @ scope_10.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(P0_1, value_91, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                T.binary(value_89, value_91, l0_1, kind=BinaryKind.ADD)
            with scope[:1, :32] as scope_11:
                value_92 = T.tensor_view(
                    T.ptr_of(qkv[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, (((cta.d1 * 2) + 0) * 512) + 256:(((cta.d1 * 2) + 0) * 512) + 256 + 1]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                g_dst0 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_93 = T.tensor_view(
                    T.ptr_of(g_dst0),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_92, value_93)
                value_94 = T.tensor_view(
                    T.ptr_of(qkv[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 1, (((cta.d1 * 2) + 1) * 512) + 256:(((cta.d1 * 2) + 1) * 512) + 256 + 1]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                g_dst1 = T.tensor_view(
                    196608,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_95 = T.tensor_view(
                    T.ptr_of(g_dst1),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_94, value_95)
            with Mesh(scope[1:2], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_12:
                value_97 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_11 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_11, 1.0)
                value_98 = T.tensor_view(
                    T.ptr_of(l0_1),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(scalar_11, value_98, value_97, kind=BinaryKind.DIV)
                value_99 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_100 = T.tensor_view(
                    T.ptr_of(value_79),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_100, value_97, value_99, kind=BinaryKind.MUL)
                ob0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_99, ob0, dtype="bf16")
                value_101 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(ob0, value_101, dtype="f32")
                value_102 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                g_dst0_2 = T.tensor_view(
                    0,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_103 = T.tensor_view(
                    T.ptr_of(g_dst0_2),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_104 = T.tensor_view(
                    T.ptr_of(value_102),
                    layout=((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.copy(value_103, value_104)
                value_105 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_104, value_105, dtype="f32")
                value_106 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_12 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_12, -1.4426950408889634)
                T.binary(value_105, scalar_12, value_106, kind=BinaryKind.MUL)
                value_107 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_106, value_107, kind=UnaryKind.EXP2)
                value_108 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_13 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_13, 1.0)
                T.binary(value_107, scalar_13, value_108, kind=BinaryKind.ADD)
                value_109 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_14 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_14, 1.0)
                T.binary(scalar_14, value_108, value_109, kind=BinaryKind.DIV)
                value_110 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_109, value_110, dtype="bf16")
                value_111 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_110, value_111, dtype="f32")
                value_112 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_101, value_111, value_112, kind=BinaryKind.MUL)
                y0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_12.d0, 2, 8 @ scope_12.d1, 32, 4 @ scope_12.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_112, y0, dtype="bf16")
                value_96 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_113 = T.tensor_view(T.ptr_of(value_96), layout=(64, 256), shape=(64, 256))
                T.copy(y0, value_113)
            m_new1_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_123 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            l1_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=256,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_13:
                value_114 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_114, 0.0)
                for o_m_6 in range(0, 64, 64):
                    for o_n_6 in range(0, 64, 64):
                        for o_k_6 in range(0, 256, 64):
                            q_dst1_3 = T.tensor_view(
                                32768,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            value_115 = T.tensor_view(
                                T.ptr_of(q_dst1_3),
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            lhs_view_15 = T.tensor_view(
                                T.ptr_of(value_115[o_m_6:o_m_6 + 64, o_k_6:o_k_6 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_116 = T.tensor_view(
                                T.ptr_of(k_ring_1_stages[0]),
                                layout=Layout(((4, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                shape=(256, 64),
                            )
                            rhs_view_15 = T.tensor_view(
                                T.ptr_of(value_116[o_k_6:o_k_6 + 16, o_n_6:o_n_6 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_114,
                                lhs_view_15,
                                rhs_view_15,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_16 = T.tensor_view(
                                T.ptr_of(value_115[o_m_6:o_m_6 + 64, o_k_6 + 16:o_k_6 + 16 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_16 = T.tensor_view(
                                T.ptr_of(value_116[o_k_6 + 16:o_k_6 + 16 + 16, o_n_6:o_n_6 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_114,
                                lhs_view_16,
                                rhs_view_16,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_17 = T.tensor_view(
                                T.ptr_of(value_115[o_m_6:o_m_6 + 64, o_k_6 + 32:o_k_6 + 32 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_17 = T.tensor_view(
                                T.ptr_of(value_116[o_k_6 + 32:o_k_6 + 32 + 16, o_n_6:o_n_6 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_114,
                                lhs_view_17,
                                rhs_view_17,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                            lhs_view_18 = T.tensor_view(
                                T.ptr_of(value_115[o_m_6:o_m_6 + 64, o_k_6 + 48:o_k_6 + 48 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            rhs_view_18 = T.tensor_view(
                                T.ptr_of(value_116[o_k_6 + 48:o_k_6 + 48 + 16, o_n_6:o_n_6 + 64]),
                                layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 64),
                            )
                            T.tiled_mma(
                                value_114,
                                lhs_view_18,
                                rhs_view_18,
                                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                            )
                S1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_117 = T.tensor_view(
                    T.ptr_of(value_114),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                value_118 = T.tensor_view(
                    T.ptr_of(value_5),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_117, value_118, S1, kind=BinaryKind.ADD)
                value_119 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(S1, value_119, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                value_120 = T.tensor_view(
                    T.ptr_of(m1),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_120, value_119, m_new1_1, kind=BinaryKind.MAX)
                value_121 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.binary(value_120, m_new1_1, value_121, kind=BinaryKind.SUB)
                value_122 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_15 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_15, 0.09016844005556021)
                T.binary(value_121, scalar_15, value_122, kind=BinaryKind.MUL)
                alpha1_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.unary(value_122, alpha1_1, kind=UnaryKind.EXP2)
                value_124 = T.tensor_view(
                    T.ptr_of(O1),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 32, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_124, alpha1_1, value_123, kind=BinaryKind.MUL)
                value_125 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 4, 2, 4 @ scope_13.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_126 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_16 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_16, 0.09016844005556021)
                T.binary(S1, scalar_16, value_126, kind=BinaryKind.MUL)
                value_127 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_17 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_17, 0.09016844005556021)
                T.binary(m_new1_1, scalar_17, value_127, kind=BinaryKind.MUL)
                value_128 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_126, value_127, value_128, kind=BinaryKind.SUB)
                P1_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_128, P1_1, kind=UnaryKind.EXP2)
                value_129 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 8, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(P1_1, value_129, dtype="bf16")
                value_130 = T.tensor_view(
                    T.ptr_of(value_125),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 4, 2, 4 @ scope_13.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_129, value_130)
                for o_m_7 in range(0, 64, 64):
                    for o_n_7 in range(0, 256, 256):
                        for o_k_7 in range(0, 64, 16):
                            value_131 = T.tensor_view(
                                T.ptr_of(value_125),
                                layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 4, 2, 4 @ scope_13.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_19 = T.tensor_view(
                                T.ptr_of(value_131[o_m_7:o_m_7 + 64, o_k_7:o_k_7 + 16]),
                                layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 2, 4 @ scope_13.d2, 2), (16, 8, 1, 512, 128, 64)),
                                shape=(64, 16),
                            )
                            value_132 = T.tensor_view(
                                T.ptr_of(v_ring_1_stages[0]),
                                layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            rhs_view_19 = T.tensor_view(
                                T.ptr_of(value_132[o_k_7:o_k_7 + 16, o_n_7:o_n_7 + 256]),
                                layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                value_123,
                                lhs_view_19,
                                rhs_view_19,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_133 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_134 = T.tensor_view(
                    T.ptr_of(l1),
                    layout=((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_134, alpha1_1, value_133, kind=BinaryKind.MUL)
                value_135 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_13.d0, 2, 8 @ scope_13.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(P1_1, value_135, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                T.binary(value_133, value_135, l1_1, kind=BinaryKind.ADD)
            with Mesh(scope[2:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_14:
                value_137 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_18 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_18, 1.0)
                value_138 = T.tensor_view(
                    T.ptr_of(l1_1),
                    layout=((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(scalar_18, value_138, value_137, kind=BinaryKind.DIV)
                value_139 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_140 = T.tensor_view(
                    T.ptr_of(value_123),
                    layout=((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_140, value_137, value_139, kind=BinaryKind.MUL)
                ob1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_139, ob1, dtype="bf16")
                value_141 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(ob1, value_141, dtype="f32")
                value_142 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                g_dst1_2 = T.tensor_view(
                    196608,
                    dtype="bf16",
                    storage="smem",
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_143 = T.tensor_view(
                    T.ptr_of(g_dst1_2),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                value_144 = T.tensor_view(
                    T.ptr_of(value_142),
                    layout=((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.copy(value_143, value_144)
                value_145 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_144, value_145, dtype="f32")
                value_146 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_19 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_19, -1.4426950408889634)
                T.binary(value_145, scalar_19, value_146, kind=BinaryKind.MUL)
                value_147 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_146, value_147, kind=UnaryKind.EXP2)
                value_148 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_20 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_20, 1.0)
                T.binary(value_147, scalar_20, value_148, kind=BinaryKind.ADD)
                value_149 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_21 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_21, 1.0)
                T.binary(scalar_21, value_148, value_149, kind=BinaryKind.DIV)
                value_150 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_149, value_150, dtype="bf16")
                value_151 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_150, value_151, dtype="f32")
                value_152 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_141, value_151, value_152, kind=BinaryKind.MUL)
                y1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_14.d0, 2, 8 @ scope_14.d1, 32, 4 @ scope_14.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_152, y1, dtype="bf16")
                value_136 = T.tensor_view(
                    32768, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_153 = T.tensor_view(
                    T.ptr_of(value_136), layout=(64, 256), shape=(64, 256)
                )
                T.copy(y1, value_153)
            with scope[:1, :32] as scope_15:
                value_96_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_155 = T.tensor_view(
                    T.ptr_of(value_96_2), layout=(64, 256), shape=(64, 256)
                )
                value_154_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
                )
                value_156 = T.tensor_view(
                    T.ptr_of(value_154_1), layout=(64, 256), shape=(64, 256)
                )
                T.copy_async_tensor(value_155, value_156)
                value_157 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 6144), shape=(1024, 6144)
                )
                window = T.tensor_view(
                    T.ptr_of(value_157[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, ((cta.d1 * 2) + 0) * 256:((cta.d1 * 2) + 0) * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_156, window)
                value_136_2 = T.tensor_view(
                    32768, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_159 = T.tensor_view(
                    T.ptr_of(value_136_2), layout=(64, 256), shape=(64, 256)
                )
                value_158_1 = T.tensor_view(
                    T.ptr_of(scratch_1[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
                )
                value_160 = T.tensor_view(
                    T.ptr_of(value_158_1), layout=(64, 256), shape=(64, 256)
                )
                T.copy_async_tensor(value_159, value_160)
                value_161 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 6144), shape=(1024, 6144)
                )
                window_1 = T.tensor_view(
                    T.ptr_of(value_161[((16 - 1) - cta.d0) * 64:((16 - 1) - cta.d0) * 64 + 64, ((cta.d1 * 2) + 1) * 256:((cta.d1 * 2) + 1) * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_160, window_1)
