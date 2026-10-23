from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(a: Tensor[(16, 32), "bf16"], b: Tensor[(32, 8), "bf16"], out: Tensor[(16, 8), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 64),), layout=(2, 32), names=("d0", "d1")) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 8, 4, 2), (64, 8, 2, 1)),
                        attrs=(S(2), S(1)),
                        mesh=Mesh((Topology("thread", 64),), layout=ComposedLayout(
                            inner=None,
                            offset=32,
                            outer=Layout((4, 8), (1, 4)),
                        ), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_1:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(512, dtype="bf16", storage="smem", layout=(16, 16), shape=(16, 16)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=(16, 16), shape=(16, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=(16, 8), shape=(16, 8)),
                T.tensor_view(256, dtype="bf16", storage="smem", layout=(16, 8), shape=(16, 8)),
            )
            for k in range(0, 32, 16):
                with scope[:1] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((16, 16), (32, 1)),
                        shape=(16, 16),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=(16, 16),
                        shape=(16, 16),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(16, 8), shape=(16, 8)
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]), layout=(16, 8), shape=(16, 8)
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_3:
                    lhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16),
                            "bf16",
                            ((2, 8 @ scope_3.d1, 2, 4 @ scope_3.d0, 2), (128, 16, 8, 2, 1)),
                            "rmem",
                        ]
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=(16, 16),
                        shape=(16, 16),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(lhs_frag_dst),
                        layout=((2, 8 @ scope_3.d1, 2, 4 @ scope_3.d0, 2), (128, 16, 8, 2, 1)),
                        shape=(16, 16),
                    )
                    T.ldmatrix(value_4, value_5)
                    rhs_frag_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 8),
                            "bf16",
                            ((2, 4 @ scope_3.d0, 2, 8 @ scope_3.d1), (64, 16, 8, 1)),
                            "rmem",
                        ]
                    )
                    value_6 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]), layout=(16, 8), shape=(16, 8)
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(rhs_frag_dst),
                        layout=((2, 4 @ scope_3.d0, 2, 8 @ scope_3.d1), (64, 16, 8, 1)),
                        shape=(16, 8),
                    )
                    T.copy(value_6, value_7)
                    for o_m in range(0, 16, 16):
                        for o_n in range(0, 8, 8):
                            for o_k in range(0, 16, 16):
                                value_8 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2, 8 @ scope_3.d1, 4 @ scope_3.d0, 2), (64, 8, 2, 1)),
                                    shape=(16, 8),
                                )
                                value_9 = T.tensor_view(
                                    T.ptr_of(lhs_frag_dst),
                                    layout=((2, 8 @ scope_3.d1, 2, 4 @ scope_3.d0, 2), (128, 16, 8, 2, 1)),
                                    shape=(16, 16),
                                )
                                value_10 = T.tensor_view(
                                    T.ptr_of(rhs_frag_dst),
                                    layout=((2, 4 @ scope_3.d0, 2, 8 @ scope_3.d1), (64, 16, 8, 1)),
                                    shape=(16, 8),
                                )
                                T.tiled_mma(value_8, value_9, value_10, atom=T.cuda.sm80.Mma())
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (16, 8),
                        "bf16",
                        ((2, 8 @ scope_4.d1, 4 @ scope_4.d0, 2), (64, 8, 2, 1)),
                        "rmem",
                    ]
                )
                value_11 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((2, 8 @ scope_4.d1, 4 @ scope_4.d0, 2), (64, 8, 2, 1)),
                    shape=(16, 8),
                )
                T.cast(value_11, result, dtype="bf16")
                value_12 = T.tensor_view(T.ptr_of(out), layout=(16, 8), shape=(16, 8))
                window = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 16, 0:0 + 8]), layout=(16, 8), shape=(16, 8)
                )
                T.copy(result, window)
