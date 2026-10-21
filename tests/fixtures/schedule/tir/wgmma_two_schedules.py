from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 32), "bf16"], c: Tensor[(64, 32), "bf16"], d: Tensor[(32, 32), "bf16"], out: Tensor[(64, 32), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 4, 4, 2), (16, 8, 1, 512, 128, 64)),
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
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(8192, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
                T.tensor_view(9216, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
            )
            lhs_2_ring_stages = (
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
                T.tensor_view(6144, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
            )
            rhs_2_ring_stages = (
                T.tensor_view(10240, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
                T.tensor_view(11264, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
            )
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(16, 32), shape=(16, 32)
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 32, 32):
                            for o_k in range(0, 16, 16):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 32),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                                    layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                    shape=(64, 16),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_5[0:0 + 64, 0:0 + 16]),
                                    layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                                    shape=(64, 16),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                                    layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                    shape=(16, 32),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[0:0 + 16, 0:0 + 32]),
                                    layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
                with scope[:1, :32] as scope_4:
                    value_7 = T.tensor_view(
                        T.ptr_of(c[0:0 + 1, k:k + 1]),
                        layout=((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(lhs_2_ring_stages[(k // 16) % 2]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    T.copy_async_tensor(value_7, value_8)
                    value_9 = T.tensor_view(
                        T.ptr_of(d[k:k + 1, 0:0 + 1]), layout=(16, 32), shape=(16, 32)
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(rhs_2_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    T.copy_async_tensor(value_9, value_10)
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    for o_m_1 in range(0, 64, 64):
                        for o_n_1 in range(0, 32, 32):
                            for o_k_1 in range(0, 16, 16):
                                value_11 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                                    shape=(64, 32),
                                )
                                value_12 = T.tensor_view(
                                    T.ptr_of(lhs_2_ring_stages[(k // 16) % 2]),
                                    layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                    shape=(64, 16),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_12[0:0 + 64, 0:0 + 16]),
                                    layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                                    shape=(64, 16),
                                )
                                value_13 = T.tensor_view(
                                    T.ptr_of(rhs_2_ring_stages[(k // 16) % 2]),
                                    layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                    shape=(16, 32),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_13[0:0 + 16, 0:0 + 32]),
                                    layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_11,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
            T.fill(out, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_6:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "bf16",
                        ((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 4, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_14 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((4 @ scope_6.d0, 2, 8 @ scope_6.d1, 4, 4 @ scope_6.d2, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.cast(value_14, result, dtype="bf16")
                value_15 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
                window = T.tensor_view(
                    T.ptr_of(value_15[0:0 + 64, 0:0 + 32]), layout=(64, 32), shape=(64, 32)
                )
                T.copy(result, window)
