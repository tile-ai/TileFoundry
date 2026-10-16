# analysis target=nvidia.h200_sxm module=SCHED_GATE_UP_T128 function=gemm topology=cta wave=121/121
# selection requested=memory executed=memory
# memory traffic=gmem:r4.10MB/w4.32MB@logical,r495.82MB/w523.23MB@total,r4.10MB/w4.32MB@cta,r4.10MB/w4.32MB@thread;rmem:r6.54MB/w6.47MB@logical,r791.45MB/w782.72MB@total,r6.54MB/w6.47MB@cta,r28.08KB/w25.89KB@thread;smem:r5.35MB/w4.10MB@logical,r647.07MB/w495.82MB@total,r5.35MB/w4.10MB@cta,r796.00KB/w4.06MB@thread footprint=a:32.00KB;v43:62:36.00KB;v45:64:4.25MB;w_gate:4.25MB;w_up:4.25MB footprint-precision=exact peak=gmem:345.85MB;rmem:216.02KB;smem:208.00KB persistent=gmem:341.56MB
#   buffer=a holds=12.83MB time=none space=cta.bn reuse=3.75MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 5120), "bf16"], w_gate: Tensor[(17424, 5120), "bf16"], w_up: Tensor[(17424, 5120), "bf16"], scratch: Tensor[(18432,), "bf16"], out: Tensor[(128, 17424), "bf16"]
):
    with Mesh((Topology("cta", 121),), layout=(1, 121), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            acc_g = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 144),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 18, 4, 2), (9216, 16, 8, 1, 512, 128, 64)),
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
                    (128, 144),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 18, 4, 2), (9216, 16, 8, 1, 512, 128, 64)),
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
            lhs_ring_stages = (
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3), shape=(128, 128)),
            )
            gate_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
                T.tensor_view(36864, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
            )
            up_ring_stages = (
                T.tensor_view(73728, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
                T.tensor_view(110592, dtype="bf16", storage="smem", layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), shape=(144, 128)),
            )
            for k in range(0, 5120, 128):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((128, 128), (5120, 1)),
                        shape=(128, 128),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 128) % 2]),
                        layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 128),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w_gate[cta.d1 * 144:cta.d1 * 144 + 1, k:k + 1]),
                        layout=((144, 128), (5120, 1)),
                        shape=(144, 128),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(gate_ring_stages[(k // 128) % 2]),
                        layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(144, 128),
                    )
                    T.copy_async_tensor(value_2, value_3)
                    value_4 = T.tensor_view(
                        T.ptr_of(w_up[cta.d1 * 144:cta.d1 * 144 + 1, k:k + 1]),
                        layout=((144, 128), (5120, 1)),
                        shape=(144, 128),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(up_ring_stages[(k // 128) % 2]),
                        layout=Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3),
                        shape=(144, 128),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 144, 144):
                            for o_k in range(0, 128, 64):
                                value_6 = T.tensor_view(
                                    T.ptr_of(acc_g),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 18, 4 @ scope_3.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 144),
                                )
                                value_7 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 128) % 2]),
                                    layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 128),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_8 = T.tensor_view(
                                    T.ptr_of(gate_ring_stages[(k // 128) % 2]),
                                    layout=Layout(((2, 64), 144), ((9216, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 144),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_8[o_k:o_k + 16, o_n:o_n + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_6,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_7[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_6,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_7[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_6,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_7[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_6,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                    for o_m_1 in range(0, 128, 128):
                        for o_n_1 in range(0, 144, 144):
                            for o_k_1 in range(0, 128, 64):
                                value_9 = T.tensor_view(
                                    T.ptr_of(acc_u),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 18, 4 @ scope_3.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 144),
                                )
                                value_10 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 128) % 2]),
                                    layout=Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 128),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_11 = T.tensor_view(
                                    T.ptr_of(up_ring_stages[(k // 128) % 2]),
                                    layout=Layout(((2, 64), 144), ((9216, 1), 64)) | Swizzle(3, 4, 3),
                                    shape=(128, 144),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_11[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_9,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 16:o_k_1 + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_5 = T.tensor_view(
                                    T.ptr_of(value_11[o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_9,
                                    lhs_view_5,
                                    rhs_view_5,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 32:o_k_1 + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_6 = T.tensor_view(
                                    T.ptr_of(value_11[o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_9,
                                    lhs_view_6,
                                    rhs_view_6,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1 + 48:o_k_1 + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_7 = T.tensor_view(
                                    T.ptr_of(value_11[o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 144]),
                                    layout=(Layout((16, (18, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 144),
                                )
                                T.tiled_mma(
                                    value_9,
                                    lhs_view_7,
                                    rhs_view_7,
                                    atom=T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                value_12 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_13 = T.tensor_view(
                    T.ptr_of(acc_g),
                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 144),
                )
                T.cast(value_13, value_12, dtype="bf16")
                g = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_12, g, dtype="f32")
                value_14 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(g, value_14, kind=UnaryKind.NEG)
                value_15 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_14, value_15, kind=UnaryKind.EXP)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(value_15, scalar, value_16, kind=BinaryKind.ADD)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(g, value_16, value_17, kind=BinaryKind.DIV)
                act = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_17, act, dtype="bf16")
                value_18 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(act, value_18, dtype="f32")
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_20 = T.tensor_view(
                    T.ptr_of(acc_u),
                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 144),
                )
                T.cast(value_20, value_19, dtype="bf16")
                u = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_19, u, dtype="f32")
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_18, u, value_21, kind=BinaryKind.MUL)
                tile_out = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_21, tile_out, dtype="bf16")
                staged_dst = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 144), shape=(128, 144)
                )
                value_22 = T.tensor_view(
                    T.ptr_of(staged_dst), layout=(128, 144), shape=(128, 144)
                )
                T.copy(tile_out, value_22)
            with scope[:1, :32] as scope_5:
                staged_dst_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 144), shape=(128, 144)
                )
                value_23 = T.tensor_view(
                    T.ptr_of(staged_dst_2), layout=(128, 144), shape=(128, 144)
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 18432]), layout=(128, 144), shape=(128, 144)
                )
                value_24 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(128, 144), shape=(128, 144)
                )
                T.copy_async_tensor(value_23, value_24)
                value_25 = T.tensor_view(T.ptr_of(out), layout=(128, 17424), shape=(128, 17424))
                window = T.tensor_view(
                    T.ptr_of(value_25[cta.d0 * 128:cta.d0 * 128 + 128, cta.d1 * 144:cta.d1 * 144 + 144]),
                    layout=((128, 144), (17424, 1)),
                    shape=(128, 144),
                )
                T.copy(value_24, window)
