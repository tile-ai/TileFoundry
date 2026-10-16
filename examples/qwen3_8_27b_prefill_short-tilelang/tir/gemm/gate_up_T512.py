# analysis target=nvidia.h200_sxm module=SCHED_GATE_UP_T512 function=gemm topology=cta wave=132/132
# selection requested=memory executed=memory
# memory traffic=gmem:r14.19MB/w17.27MB@logical,r1.83GB/w2.23GB@total,r14.19MB/w17.27MB@cta,r14.19MB/w17.27MB@thread;rmem:r44.61MB/w44.00MB@logical,r5.75GB/w5.72GB@total,r44.61MB/w44.34MB@cta,r189.90KB/w177.38KB@thread;smem:r17.94MB/w14.19MB@logical,r2.31GB/w1.83GB@total,r17.94MB/w14.19MB@cta,r2.82MB/w14.06MB@thread footprint=a:64.00KB;v45:83:44.00KB;v47:84:5.67MB;w_gate:726.00KB;w_up:726.00KB footprint-precision=exact peak=gmem:362.33MB;rmem:264.00KB;smem:224.00KB persistent=gmem:345.31MB
#   error="rmem placement peak 264.00KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(512, 5120), "bf16"], w_gate: Tensor[(17424, 5120), "bf16"], w_up: Tensor[(17424, 5120), "bf16"], out: Tensor[(512, 17424), "bf16"]
):
    with Mesh((Topology("cta", 132),), layout=(132,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(135168, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(151552, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(167936, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            gate_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
                T.tensor_view(22528, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
                T.tensor_view(45056, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
            )
            up_ring_stages = (
                T.tensor_view(67584, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
                T.tensor_view(90112, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
                T.tensor_view(112640, dtype="bf16", storage="smem", layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(176, 64)),
            )
            for u in range(cta.d0, 396, 132):
                acc_g = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 176),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 22, 4, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            attrs=(S(0), S(1), S(3), S(5)),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                            ), names=("d0", "d1", "d2", "d3")),
                        ),
                        "rmem",
                    ]
                )
                acc_u = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 176),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 22, 4, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            attrs=(S(0), S(1), S(3), S(5)),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                            ), names=("d0", "d1", "d2", "d3")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_1:
                    T.fill(acc_g, 0.0)
                    T.fill(acc_u, 0.0)
                for k in range(0, 5120, 64):
                    with scope[:1, :32] as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(a[(u % 4) * 128:(u % 4) * 128 + 1, k:k + 1]),
                            layout=((128, 64), (5120, 1)),
                            shape=(128, 64),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                            layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 64),
                        )
                        T.copy_async_tensor(value, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(w_gate[(u // 4) * 176:(u // 4) * 176 + 1, k:k + 1]),
                            layout=((176, 64), (5120, 1)),
                            shape=(176, 64),
                        )
                        value_3 = T.tensor_view(
                            T.ptr_of(gate_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                            layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(176, 64),
                        )
                        T.copy_async_tensor(value_2, value_3)
                        value_4 = T.tensor_view(
                            T.ptr_of(w_up[(u // 4) * 176:(u // 4) * 176 + 1, k:k + 1]),
                            layout=((176, 64), (5120, 1)),
                            shape=(176, 64),
                        )
                        value_5 = T.tensor_view(
                            T.ptr_of(up_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                            layout=Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                            shape=(176, 64),
                        )
                        T.copy_async_tensor(value_4, value_5)
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        for o_m in range(0, 128, 128):
                            for o_n in range(0, 176, 176):
                                for o_k in range(0, 64, 64):
                                    value_6 = T.tensor_view(
                                        T.ptr_of(acc_g),
                                        layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 22, 4 @ scope_3.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                                        shape=(128, 176),
                                    )
                                    value_7 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 64),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    value_8 = T.tensor_view(
                                        T.ptr_of(gate_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=((64, 176), (1, 64)) | Swizzle(3, 4, 3),
                                        shape=(64, 176),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_8[o_k:o_k + 16, o_n:o_n + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_6,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_7[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(value_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_6,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_7[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(value_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_6,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_7[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(value_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_6,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                        for o_m_1 in range(0, 128, 128):
                            for o_n_1 in range(0, 176, 176):
                                for o_k_1 in range(0, 64, 64):
                                    value_9 = T.tensor_view(
                                        T.ptr_of(acc_u),
                                        layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 22, 4 @ scope_3.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                                        shape=(128, 176),
                                    )
                                    value_10 = T.tensor_view(
                                        T.ptr_of(lhs_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                        shape=(128, 64),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    value_11 = T.tensor_view(
                                        T.ptr_of(up_ring_stages[((((u - cta.d0) // 132) * 80) + (k // 64)) % 3]),
                                        layout=((64, 176), (1, 64)) | Swizzle(3, 4, 3),
                                        shape=(64, 176),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(value_11[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_9,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 16:o_k_1 + 16 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(value_11[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_9,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 32:o_k_1 + 32 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(value_11[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_9,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 48:o_k_1 + 48 + 16]),
                                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(value_11[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 176]),
                                        layout=(Layout((16, (22, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                        shape=(16, 176),
                                    )
                                    T.tiled_mma(
                                        value_9,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                u_2 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 176),
                        "f32",
                        ShardLayout(
                            layout=Layout((2, 4, 2, 8, 22, 4, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            attrs=(S(0), S(1), S(3), S(5)),
                            mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                                inner=None,
                                offset=128,
                                outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                            ), names=("d0", "d1", "d2", "d3")),
                        ),
                        "rmem",
                    ]
                )
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    value_12 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(acc_u),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 176),
                    )
                    T.cast(value_13, value_12, dtype="bf16")
                    T.cast(value_12, u_2, dtype="f32")
                    value_15 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(acc_g),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 176),
                    )
                    T.cast(value_16, value_15, dtype="bf16")
                    g = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_15, g, dtype="f32")
                    value_17 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(g, value_17, kind=UnaryKind.NEG)
                    value_18 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.unary(value_17, value_18, kind=UnaryKind.EXP)
                    value_19 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                    T.fill(scalar, 1.0)
                    T.binary(value_18, scalar, value_19, kind=BinaryKind.ADD)
                    value_20 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(g, value_19, value_20, kind=BinaryKind.DIV)
                    act = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_20, act, dtype="bf16")
                    value_21 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(act, value_21, dtype="f32")
                    value_22 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "f32",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.binary(value_21, u_2, value_22, kind=BinaryKind.MUL)
                    tile_out = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 176),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 22, 4 @ scope_4.d3, 2), (11264, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    T.cast(value_22, tile_out, dtype="bf16")
                    value_14 = T.tensor_view(
                        184320,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 176),
                        shape=(128, 176),
                    )
                    value_23 = T.tensor_view(
                        T.ptr_of(value_14), layout=(128, 176), shape=(128, 176)
                    )
                    T.copy(tile_out, value_23)
                with scope[1:2, :32] as scope_5:
                    value_14_2 = T.tensor_view(
                        184320,
                        dtype="bf16",
                        storage="smem",
                        layout=(128, 176),
                        shape=(128, 176),
                    )
                    value_24 = T.tensor_view(
                        T.ptr_of(value_14_2), layout=(128, 176), shape=(128, 176)
                    )
                    value_25 = T.tensor_view(
                        T.ptr_of(out[(128 * u) - (512 * (u // 4)):(128 * u) - (512 * (u // 4)) + 1, (176 * u) - (176 * (((3 * u) + 3) // 4)):(176 * u) - (176 * (((3 * u) + 3) // 4)) + 1]),
                        layout=((128, 176), (17424, 1)),
                        shape=(128, 176),
                    )
                    T.copy_async_tensor(value_24, value_25)
