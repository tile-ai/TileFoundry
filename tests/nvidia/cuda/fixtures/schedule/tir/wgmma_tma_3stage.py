from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384),),
)
class WgmmaTma3stage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        a: Tensor[(128, 64), "bf16"], b: Tensor[(64, 16), "bf16"], out: Tensor[(128, 16), "bf16"]
    ):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
                acc = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 16),
                        "f32",
                        ShardLayout(
                        layout=Layout((2, 4, 2, 8, 2, 4, 2), (1024, 16, 8, 1, 512, 128, 64)),
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
                    T.fill(acc, 0.0)
                for k in range(0, 64, 16):
                    with scope[:1, :32] as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(a[0:0 + 1, k:k + 1]),
                            layout=((128, 16), (64, 1)),
                            shape=(128, 16),
                        )
                        lhs_ring_1 = T.tensor_view(
                            ((4096 * (k // 16)) - (12288 * (k // 48))) + 1536,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                            shape=(128, 16),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_1),
                            layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                            shape=(128, 16),
                        )
                        T.copy_async_tensor(value, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(16, 16), shape=(16, 16)
                        )
                        rhs_ring_1 = T.tensor_view(
                            (512 * (k // 16)) - (1536 * (k // 48)),
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                            shape=(16, 16),
                        )
                        value_3 = T.tensor_view(
                            T.ptr_of(rhs_ring_1),
                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                            shape=(16, 16),
                        )
                        T.copy_async_tensor(value_2, value_3)
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_3:
                        for o_m in range(0, 128, 128):
                            for o_n in range(0, 16, 16):
                                for o_k in range(0, 16, 16):
                                    value_4 = T.tensor_view(
                                        T.ptr_of(acc),
                                        layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 2, 4 @ scope_3.d3, 2), (1024, 16, 8, 1, 512, 128, 64)),
                                        shape=(128, 16),
                                    )
                                    lhs_ring_2 = T.tensor_view(
                                        ((4096 * (k // 16)) - (12288 * (k // 48))) + 1536,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                                        shape=(128, 16),
                                    )
                                    value_5 = T.tensor_view(
                                        T.ptr_of(lhs_ring_2),
                                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                                        shape=(128, 16),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_5[0:0 + 128, 0:0 + 16]),
                                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                                        shape=(128, 16),
                                    )
                                    rhs_ring_2 = T.tensor_view(
                                        (512 * (k // 16)) - (1536 * (k // 48)),
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                        shape=(16, 16),
                                    )
                                    value_6 = T.tensor_view(
                                        T.ptr_of(rhs_ring_2),
                                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                        shape=(16, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_6[0:0 + 16, 0:0 + 16]),
                                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                    )
                T.fill(out, 0.0)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_4:
                    result = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 16),
                            "bf16",
                            ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 2, 4 @ scope_4.d3, 2), (1024, 16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 2, 4 @ scope_4.d3, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    T.cast(value_7, result, dtype="bf16")
                    value_8 = T.tensor_view(T.ptr_of(out), layout=(128, 16), shape=(128, 16))
                    window = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 128, 0:0 + 16]), layout=(128, 16), shape=(128, 16)
                    )
                    T.copy(result, window)
