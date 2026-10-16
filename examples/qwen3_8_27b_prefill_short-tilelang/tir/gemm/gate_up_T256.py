# analysis target=nvidia.h200_sxm module=SCHED_GATE_UP_T256 function=gemm topology=cta wave=132/242
# selection requested=memory executed=memory
# memory traffic=gmem:r4.10MB/w8.58MB@logical,r991.63MB/w2.03GB@total,r4.10MB/w8.58MB@cta,r4.10MB/w8.58MB@thread;rmem:r12.17MB/w12.09MB@logical,r2.88GB/w2.86GB@total,r12.17MB/w12.09MB@cta,r52.46KB/w48.39KB@thread;smem:r5.35MB/w4.10MB@logical,r1.26GB/w991.63MB@total,r5.35MB/w4.10MB@cta,r796.00KB/w4.06MB@thread footprint=a:32.00KB;v43:77:36.00KB;v45:78:4.64MB;w_gate:1.16MB;w_up:1.16MB footprint-precision=exact peak=gmem:351.36MB;rmem:216.02KB;smem:208.00KB persistent=gmem:342.81MB
#   buffer=a holds=7.03MB time=none space=cta.bn reuse=2.03MB fits=yes precision=exact
#   buffer=w_gate holds=7.03MB time=none space=cta.bm reuse=1.16MB fits=yes precision=exact
#   buffer=w_up holds=7.03MB time=none space=cta.bm reuse=1.16MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 5120), "bf16"], w_gate: Tensor[(17424, 5120), "bf16"], w_up: Tensor[(17424, 5120), "bf16"], scratch: Tensor[(18432,), "bf16"], out: Tensor[(256, 17424), "bf16"]
):
    with Mesh((Topology("cta", 242),), layout=(121, 2), names=("d0", "d1")) as cta:
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
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(196608, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            gate_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(18432, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(36864, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(55296, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
            )
            up_ring_stages = (
                T.tensor_view(73728, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(92160, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(110592, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
                T.tensor_view(129024, dtype="bf16", storage="smem", layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(144, 64)),
            )
            for k in range(0, 5120, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 1, k:k + 1]),
                        layout=((128, 64), (5120, 1)),
                        shape=(128, 64),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 4]),
                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w_gate[cta.d0 * 144:cta.d0 * 144 + 1, k:k + 1]),
                        layout=((144, 64), (5120, 1)),
                        shape=(144, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(gate_ring_stages[(k // 64) % 4]),
                        layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(144, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                    value_4 = T.tensor_view(
                        T.ptr_of(w_up[cta.d0 * 144:cta.d0 * 144 + 1, k:k + 1]),
                        layout=((144, 64), (5120, 1)),
                        shape=(144, 64),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(up_ring_stages[(k // 64) % 4]),
                        layout=Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(144, 64),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 144, 144):
                            for o_k in range(0, 64, 64):
                                value_6 = T.tensor_view(
                                    T.ptr_of(acc_g),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 18, 4 @ scope_3.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 144),
                                )
                                value_7 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 64) % 4]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_7[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_8 = T.tensor_view(
                                    T.ptr_of(gate_ring_stages[(k // 64) % 4]),
                                    layout=((64, 144), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 144),
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
                            for o_k_1 in range(0, 64, 64):
                                value_9 = T.tensor_view(
                                    T.ptr_of(acc_u),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 18, 4 @ scope_3.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 144),
                                )
                                value_10 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 64) % 4]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(value_10[o_m_1:o_m_1 + 128, o_k_1:o_k_1 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_11 = T.tensor_view(
                                    T.ptr_of(up_ring_stages[(k // 64) % 4]),
                                    layout=((64, 144), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 144),
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
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_14 = T.tensor_view(
                    T.ptr_of(acc_g),
                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 144),
                )
                T.cast(value_14, value_13, dtype="bf16")
                g = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_13, g, dtype="f32")
                value_15 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(g, value_15, kind=UnaryKind.NEG)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.unary(value_15, value_16, kind=UnaryKind.EXP)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                T.binary(value_16, scalar, value_17, kind=BinaryKind.ADD)
                value_18 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(g, value_17, value_18, kind=BinaryKind.DIV)
                act = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_18, act, dtype="bf16")
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(act, value_19, dtype="f32")
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_21 = T.tensor_view(
                    T.ptr_of(acc_u),
                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 144),
                )
                T.cast(value_21, value_20, dtype="bf16")
                u = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_20, u, dtype="f32")
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "f32",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.binary(value_19, u, value_22, kind=BinaryKind.MUL)
                tile_out = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 144),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 18, 4 @ scope_4.d3, 2), (9216, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(value_22, tile_out, dtype="bf16")
                value_12 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 144), shape=(128, 144)
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_12), layout=(128, 144), shape=(128, 144)
                )
                T.copy(tile_out, value_23)
            with scope[1:2, :32] as scope_5:
                value_12_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 144), shape=(128, 144)
                )
                value_25 = T.tensor_view(
                    T.ptr_of(value_12_2), layout=(128, 144), shape=(128, 144)
                )
                value_24_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 18432]), layout=(128, 144), shape=(128, 144)
                )
                value_26 = T.tensor_view(
                    T.ptr_of(value_24_1), layout=(128, 144), shape=(128, 144)
                )
                T.copy_async_tensor(value_25, value_26)
                value_27 = T.tensor_view(T.ptr_of(out), layout=(256, 17424), shape=(256, 17424))
                window = T.tensor_view(
                    T.ptr_of(value_27[cta.d1 * 128:cta.d1 * 128 + 128, cta.d0 * 144:cta.d0 * 144 + 144]),
                    layout=((128, 144), (17424, 1)),
                    shape=(128, 144),
                )
                T.copy(value_26, window)
