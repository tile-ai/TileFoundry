from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256),),
)
class WgmmaAKMajor:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        at: Tensor[(64, 64), "fp8e4m3"], b: Tensor[(64, 32), "fp8e4m3", ((64, 32), (1, 64))], scratch: Tensor[(2048,), "fp8e4m3"], out: Tensor[(64, 32), "bf16"]
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
                lhs_ring_0 = T.tensor_view(
                    2048,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                    shape=(64, 32),
                )
                lhs_ring_1 = T.tensor_view(
                    4096,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                    shape=(64, 32),
                )
                lhs_ring_stages = (
                lhs_ring_0,
                lhs_ring_1,
            )
                rhs_ring_0 = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                    shape=(32, 32),
                )
                rhs_ring_1 = T.tensor_view(
                    1024,
                    dtype="fp8e4m3",
                    storage="smem",
                    layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                    shape=(32, 32),
                )
                rhs_ring_stages = (
                rhs_ring_0,
                rhs_ring_1,
            )
                for k in range(0, 64, 32):
                    with scope[:1, :32] as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(at[k:k + 1, 0:0 + 1]), layout=(32, 64), shape=(32, 64)
                        )
                        tile = T.tensor_view(
                            T.ptr_of(value[0:0 + 32, 0:0 + 64]),
                            layout=((64, 32), (1, 64)),
                            shape=(64, 32),
                        )
                        a_1 = T.tensor_view(
                            T.ptr_of(scratch[0:0 + 2048]), layout=(64, 32), shape=(64, 32)
                        )
                        T.copy(tile, a_1)
                        value_1 = lhs_ring_stages[(k // 32) % 2]
                        T.copy_async_tensor(a_1, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(b[k:k + 1, 0:0 + 1]),
                            layout=((32, 32), (1, 64)),
                            shape=(32, 32),
                        )
                        value_3 = rhs_ring_stages[(k // 32) % 2]
                        T.copy_async_tensor(value_2, value_3)
                    with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 32, 32):
                                    value_4 = T.tensor_view(
                                        T.ptr_of(acc),
                                        layout=((4 @ scope_3.d0, 2, 8 @ scope_3.d1, 4, 4 @ scope_3.d2, 2), (16, 8, 1, 512, 128, 64)),
                                        shape=(64, 32),
                                    )
                                    value_5 = lhs_ring_stages[(k // 32) % 2]
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_5[0:0 + 64, 0:0 + 32]),
                                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))) @ {B()},
                                        shape=(64, 32),
                                    )
                                    value_6 = rhs_ring_stages[(k // 32) % 2]
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_6[0:0 + 32, 0:0 + 32]),
                                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))) @ {B()},
                                        shape=(32, 32),
                                    )
                                    T.tiled_mma(
                                        value_4,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                    )
                T.fill(out, 0.0)
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_4:
                    result = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 32),
                            "bf16",
                            ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 4, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 4, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    T.cast(value_7, result, dtype="bf16")
                    value_8 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
                    window = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 64, 0:0 + 32]), layout=(64, 32), shape=(64, 32)
                    )
                    T.copy(result, window)
