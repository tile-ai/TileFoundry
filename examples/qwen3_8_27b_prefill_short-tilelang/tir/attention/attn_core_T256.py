# analysis target=nvidia.h200_sxm module=SCHED_ATTN_CORE_T256 function=attn_core topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r368.00KB/w3.06MB@logical,r34.50MB/w294.00MB@total,r368.00KB/w3.06MB@cta,r352.12KB/w3.06MB@thread;rmem:r1.86MB/w1.68MB@logical,r178.33MB/w164.44MB@total,r1.86MB/w1.71MB@cta,r18.57KB/w16.97KB@thread;smem:r480.00KB/w384.00KB@logical,r45.00MB/w36.00MB@total,r480.00KB/w384.00KB@cta,r76.75KB/w320.75KB@thread footprint=diag:16.00KB;k:512.00KB;q:3.00MB;qkv:3.00MB;v:512.00KB;v120:111:32.00KB;v124:112:3.00MB footprint-precision=exact peak=gmem:14.05MB;rmem:160.27KB;smem:160.00KB persistent=gmem:11.02MB
#   buffer=k holds=10.05MB time=none space=cta.blk reuse=1.50MB fits=yes precision=exact
#   buffer=v holds=10.05MB time=none space=cta.blk reuse=1.50MB fits=yes precision=exact
#   buffer=diag holds=10.05MB time=none space=cta.blk,cta.hg reuse=1.48MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def attn_core(
    q: Tensor[(256, 24, 256), "bf16"], k: Tensor[(256, 4, 256), "bf16"], v: Tensor[(256, 4, 256), "bf16"], qkv: Tensor[(256, 14336), "bf16"], diag: Tensor[(64, 64), "f32"], scratch: Tensor[(16384,), "bf16"], out: Tensor[(256, 6144), "bf16"]
):
    k_ring_1_stages = (
        T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
    )
    v_ring_1_stages = (
        T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
        T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
    )
    with Mesh((Topology("cta", 96),), layout=(4, 24), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            O0 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
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
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
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
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
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
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_1:
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
            with scope[:1, :32] as scope_2:
                value_4 = T.tensor_view(
                    T.ptr_of(q[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 1, cta.d1:cta.d1 + 1, 0:0 + 1]),
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
                value_5 = T.tensor_view(
                    T.ptr_of(q_dst0),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_4, value_5)
            k_ring_stages = (
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            v_ring_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for j in range(0, (0 - cta.d0) + 3, 1):
                with scope[:1, :32] as scope_3:
                    value_6 = T.tensor_view(
                        T.ptr_of(k[j * 64:j * 64 + 1, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 1]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(k_ring_stages[j % 2]),
                        layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 256),
                    )
                    T.copy_async_tensor(value_6, value_7)
                with scope[:1, 32:64] as scope_4:
                    value_8 = T.tensor_view(
                        T.ptr_of(v[j * 64:j * 64 + 1, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 1]),
                        layout=((64, 256), (1024, 1)),
                        shape=(64, 256),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(v_ring_stages[j % 2]),
                        layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                        shape=(64, 256),
                    )
                    T.copy_async_tensor(value_8, value_9)
                m_new0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ShardLayout(
                            layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            attrs=(S(0), S(2), B()),
                            mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((4, 8, 4), (32, 4, 1)),
                            ), names=("d0", "d1", "d2")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    value_10 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.fill(value_10, 0.0)
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 64, 64):
                            for o_k in range(0, 256, 64):
                                q_dst0_2 = T.tensor_view(
                                    0,
                                    dtype="bf16",
                                    storage="smem",
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                value_11 = T.tensor_view(
                                    T.ptr_of(q_dst0_2),
                                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_11[o_m:o_m + 64, o_k:o_k + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                value_12 = T.tensor_view(
                                    T.ptr_of(k_ring_stages[j % 2]),
                                    layout=Layout(((4, 64), 64), ((4096, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(256, 64),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_12[o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_10,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_11[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_12[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_10,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_11[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_12[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_10,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_11[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((8, 8), 16), ((512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(64, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_12[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                    layout=(Layout((16, (8, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    value_10,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    value_13 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_10),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.reduce(value_14, value_13, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                    value_15 = T.tensor_view(
                        T.ptr_of(m0),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_15, value_13, m_new0, kind=BinaryKind.MAX)
                    value_16 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_15, m_new0, value_16, kind=BinaryKind.SUB)
                    value_17 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_1, 0.09016844005556021)
                    T.binary(value_16, scalar_1, value_17, kind=BinaryKind.MUL)
                    alpha0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.unary(value_17, alpha0, kind=UnaryKind.EXP2)
                    value_18 = T.tensor_view(
                        T.ptr_of(O0),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 32, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    value_19 = T.tensor_view(
                        T.ptr_of(O0),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 32, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 256),
                    )
                    T.binary(value_18, alpha0, value_19, kind=BinaryKind.MUL)
                    value_20 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 2, 4 @ scope_5.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_21 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_2, 0.09016844005556021)
                    T.binary(value_14, scalar_2, value_21, kind=BinaryKind.MUL)
                    value_22 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar_3, 0.09016844005556021)
                    T.binary(m_new0, scalar_3, value_22, kind=BinaryKind.MUL)
                    value_23 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_21, value_22, value_23, kind=BinaryKind.SUB)
                    P0 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_23, P0, kind=UnaryKind.EXP2)
                    value_24 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 64),
                            "bf16",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 8, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(P0, value_24, dtype="bf16")
                    value_25 = T.tensor_view(
                        T.ptr_of(value_20),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 2, 4 @ scope_5.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        shape=(64, 64),
                    )
                    T.copy(value_24, value_25)
                    for o_m_1 in range(0, 64, 64):
                        for o_n_1 in range(0, 256, 256):
                            for o_k_1 in range(0, 64, 16):
                                value_26 = T.tensor_view(
                                    T.ptr_of(O0),
                                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 32, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 256),
                                )
                                value_27 = T.tensor_view(
                                    T.ptr_of(value_20),
                                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 2, 4 @ scope_5.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                    shape=(64, 64),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_27[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 2, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 16),
                                )
                                value_28 = T.tensor_view(
                                    T.ptr_of(v_ring_stages[j % 2]),
                                    layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_28[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                    layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_26,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                    value_29 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_30 = T.tensor_view(
                        T.ptr_of(l0),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_30, alpha0, value_29, kind=BinaryKind.MUL)
                    value_31 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(P0, value_31, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                    value_32 = T.tensor_view(
                        T.ptr_of(l0),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(value_29, value_31, value_32, kind=BinaryKind.ADD)
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                    value_33 = T.tensor_view(
                        T.ptr_of(m_new0),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    value_34 = T.tensor_view(
                        T.ptr_of(m0),
                        layout=((4 @ threads.d0, 2, 8 @ threads.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.copy(value_33, value_34)
            with scope[:1, :32] as scope_6:
                value_35 = T.tensor_view(
                    T.ptr_of(k[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 1, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 1]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                value_36 = T.tensor_view(
                    T.ptr_of(k_ring_1_stages[0]),
                    layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_35, value_36)
            with scope[:1, 32:64] as scope_7:
                value_37 = T.tensor_view(
                    T.ptr_of(v[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 1, cta.d1 // 6:cta.d1 // 6 + 1, 0:0 + 1]),
                    layout=((64, 256), (1024, 1)),
                    shape=(64, 256),
                )
                value_38 = T.tensor_view(
                    T.ptr_of(v_ring_1_stages[0]),
                    layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                    shape=(64, 256),
                )
                T.copy_async_tensor(value_37, value_38)
            m_new0_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 1),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        attrs=(S(0), S(2), B()),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            value_48 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 32, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
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
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("d0", "d1", "d2")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_8:
                value_39 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.fill(value_39, 0.0)
                for o_m_2 in range(0, 64, 64):
                    for o_n_2 in range(0, 64, 64):
                        for o_k_2 in range(0, 256, 64):
                            q_dst0_3 = T.tensor_view(
                                0,
                                dtype="bf16",
                                storage="smem",
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            value_40 = T.tensor_view(
                                T.ptr_of(q_dst0_3),
                                layout=Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            lhs_view_5 = T.tensor_view(
                                T.ptr_of(value_40[o_m_2:o_m_2 + 64, o_k_2:o_k_2 + 16]),
                                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(64, 16),
                            )
                            value_41 = T.tensor_view(
                                T.ptr_of(k_ring_1_stages[0]),
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
                S0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_42 = T.tensor_view(
                    T.ptr_of(value_39),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                value_43 = T.tensor_view(
                    T.ptr_of(value_1),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.binary(value_42, value_43, S0, kind=BinaryKind.ADD)
                value_44 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(S0, value_44, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
                value_45 = T.tensor_view(
                    T.ptr_of(m0),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_45, value_44, m_new0_1, kind=BinaryKind.MAX)
                value_46 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.binary(value_45, m_new0_1, value_46, kind=BinaryKind.SUB)
                value_47 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_4, 0.09016844005556021)
                T.binary(value_46, scalar_4, value_47, kind=BinaryKind.MUL)
                alpha0_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.unary(value_47, alpha0_1, kind=UnaryKind.EXP2)
                value_49 = T.tensor_view(
                    T.ptr_of(O0),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 32, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_49, alpha0_1, value_48, kind=BinaryKind.MUL)
                value_50 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 4, 2, 4 @ scope_8.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_51 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_5, 0.09016844005556021)
                T.binary(S0, scalar_5, value_51, kind=BinaryKind.MUL)
                value_52 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_6, 0.09016844005556021)
                T.binary(m_new0_1, scalar_6, value_52, kind=BinaryKind.MUL)
                value_53 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_51, value_52, value_53, kind=BinaryKind.SUB)
                P0_1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_53, P0_1, kind=UnaryKind.EXP2)
                value_54 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64),
                        "bf16",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 8, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(P0_1, value_54, dtype="bf16")
                value_55 = T.tensor_view(
                    T.ptr_of(value_50),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 4, 2, 4 @ scope_8.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                    shape=(64, 64),
                )
                T.copy(value_54, value_55)
                for o_m_3 in range(0, 64, 64):
                    for o_n_3 in range(0, 256, 256):
                        for o_k_3 in range(0, 64, 16):
                            value_56 = T.tensor_view(
                                T.ptr_of(value_50),
                                layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 4, 2, 4 @ scope_8.d2, 2), (16, 8, 1, 1024, 512, 128, 64)),
                                shape=(64, 64),
                            )
                            lhs_view_9 = T.tensor_view(
                                T.ptr_of(value_56[o_m_3:o_m_3 + 64, o_k_3:o_k_3 + 16]),
                                layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 2, 4 @ scope_8.d2, 2), (16, 8, 1, 512, 128, 64)),
                                shape=(64, 16),
                            )
                            value_57 = T.tensor_view(
                                T.ptr_of(v_ring_1_stages[0]),
                                layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3),
                                shape=(64, 256),
                            )
                            rhs_view_9 = T.tensor_view(
                                T.ptr_of(value_57[o_k_3:o_k_3 + 16, o_n_3:o_n_3 + 256]),
                                layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                shape=(16, 256),
                            )
                            T.tiled_mma(
                                value_48,
                                lhs_view_9,
                                rhs_view_9,
                                atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                            )
                value_58 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_59 = T.tensor_view(
                    T.ptr_of(l0),
                    layout=((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(value_59, alpha0_1, value_58, kind=BinaryKind.MUL)
                value_60 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_8.d0, 2, 8 @ scope_8.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(P0_1, value_60, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                T.binary(value_58, value_60, l0_1, kind=BinaryKind.ADD)
            with scope[:1, 32:64] as scope_9:
                value_61 = T.tensor_view(
                    T.ptr_of(qkv[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 1, (cta.d1 * 512) + 256:(cta.d1 * 512) + 256 + 1]),
                    layout=((64, 256), (14336, 1)),
                    shape=(64, 256),
                )
                g_dst = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_62 = T.tensor_view(T.ptr_of(g_dst), layout=(64, 256), shape=(64, 256))
                T.copy_async_tensor(value_61, value_62)
            with Mesh(scope[:1, 64:], layout=(64,), names=("d0",)) as scope_10:
                value_64 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "bf16", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                g_dst_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_65 = T.tensor_view(T.ptr_of(g_dst_2), layout=(64, 256), shape=(64, 256))
                value_66 = T.tensor_view(
                    T.ptr_of(value_64),
                    layout=((64 @ scope_10.d0, 256), (0, 1)),
                    shape=(64, 256),
                )
                T.copy(value_65, value_66)
                value_67 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "f32", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                T.cast(value_66, value_67, dtype="f32")
                value_68 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "f32", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                scalar_7 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_7, -1.4426950408889634)
                T.binary(value_67, scalar_7, value_68, kind=BinaryKind.MUL)
                value_69 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "f32", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                T.unary(value_68, value_69, kind=UnaryKind.EXP2)
                value_70 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "f32", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                scalar_8 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_8, 1.0)
                T.binary(value_69, scalar_8, value_70, kind=BinaryKind.ADD)
                value_71 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "f32", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                scalar_9 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_9, 1.0)
                T.binary(scalar_9, value_70, value_71, kind=BinaryKind.DIV)
                sig = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256), "bf16", ((64 @ scope_10.d0, 256), (0, 1)), "rmem"
                    ]
                )
                T.cast(value_71, sig, dtype="bf16")
                value_63 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_72 = T.tensor_view(T.ptr_of(value_63), layout=(64, 256), shape=(64, 256))
                T.copy(sig, value_72)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_11:
                value_74 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                scalar_10 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_10, 1.0)
                value_75 = T.tensor_view(
                    T.ptr_of(l0_1),
                    layout=((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.binary(scalar_10, value_75, value_74, kind=BinaryKind.DIV)
                value_76 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_77 = T.tensor_view(
                    T.ptr_of(value_48),
                    layout=((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.binary(value_77, value_74, value_76, kind=BinaryKind.MUL)
                ob0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_76, ob0, dtype="bf16")
                value_78 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(ob0, value_78, dtype="f32")
                value_79 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_63_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_80 = T.tensor_view(
                    T.ptr_of(value_63_2), layout=(64, 256), shape=(64, 256)
                )
                value_81 = T.tensor_view(
                    T.ptr_of(value_79),
                    layout=((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 256),
                )
                T.copy(value_80, value_81)
                value_82 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_81, value_82, dtype="f32")
                value_83 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "f32",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_78, value_82, value_83, kind=BinaryKind.MUL)
                y0 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 256),
                        "bf16",
                        ((4 @ scope_11.d0, 2, 8 @ scope_11.d1, 32, 4 @ scope_11.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_83, y0, dtype="bf16")
                value_73 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_84 = T.tensor_view(T.ptr_of(value_73), layout=(64, 256), shape=(64, 256))
                T.copy(y0, value_84)
            with scope[:1, :32] as scope_12:
                value_73_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(64, 256), shape=(64, 256)
                )
                value_86 = T.tensor_view(
                    T.ptr_of(value_73_2), layout=(64, 256), shape=(64, 256)
                )
                value_85_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 16384]), layout=(64, 256), shape=(64, 256)
                )
                value_87 = T.tensor_view(
                    T.ptr_of(value_85_1), layout=(64, 256), shape=(64, 256)
                )
                T.copy_async_tensor(value_86, value_87)
                value_88 = T.tensor_view(T.ptr_of(out), layout=(256, 6144), shape=(256, 6144))
                window = T.tensor_view(
                    T.ptr_of(value_88[((4 - 1) - cta.d0) * 64:((4 - 1) - cta.d0) * 64 + 64, cta.d1 * 256:cta.d1 * 256 + 256]),
                    layout=((64, 256), (6144, 1)),
                    shape=(64, 256),
                )
                T.copy(value_87, window)
