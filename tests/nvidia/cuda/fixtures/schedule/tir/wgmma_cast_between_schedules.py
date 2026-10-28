from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256),),
)
class WgmmaCastBetweenSchedules:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(
        a: Tensor[(64, 32), "bf16"], b_f32: Tensor[(32, 32), "f32"], bias: Tensor[(64, 32), "f32"], out: Tensor[(64, 1), "bf16"]
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
                for k in range(0, 32, 16):
                    with Mesh((Topology("thread", 256),), layout=(32,), names=("d0",)) as scope_2:
                        value = T.tensor_view(
                            T.ptr_of(a[0:0 + 1, k:k + 1]),
                            layout=((64, 16), (32, 1)),
                            shape=(64, 16),
                        )
                        lhs_ring_1 = T.tensor_view(
                            ((2048 * (k // 16)) - (4096 * (k // 32))) + 2048,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                            shape=(64, 16),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(lhs_ring_1),
                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                            shape=(64, 16),
                        )
                        T.copy_async_tensor(value, value_1)
                    with Mesh((Topology("thread", 256),), layout=(32,), names=("d0",)) as scope_3:
                        b_tile_dst = T.alloc_tensor(
                            tensor_type=Tensor[
                                (16, 32), "f32", ((16, 32 @ scope_3.d0), (1, 16)), "rmem"
                            ]
                        )
                        value_2 = T.tensor_view(
                            T.ptr_of(b_f32[k:k + 1, 0:0 + 1]), layout=(16, 32), shape=(16, 32)
                        )
                        value_3 = T.tensor_view(
                            T.ptr_of(b_tile_dst),
                            layout=((16, 32 @ scope_3.d0), (1, 16)),
                            shape=(16, 32),
                        )
                        T.copy(value_2, value_3)
                        b = T.alloc_tensor(
                            tensor_type=Tensor[
                                (16, 32), "bf16", ((16, 32 @ scope_3.d0), (1, 16)), "rmem"
                            ]
                        )
                        T.cast(value_3, b, dtype="bf16")
                        rhs_ring_1 = T.tensor_view(
                            (1024 * (k // 16)) - (2048 * (k // 32)),
                            dtype="bf16",
                            storage="smem",
                            layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                            shape=(16, 32),
                        )
                        value_4 = T.tensor_view(
                            T.ptr_of(rhs_ring_1),
                            layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                            shape=(16, 32),
                        )
                        T.copy(b, value_4)
                    with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_4:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 16, 16):
                                    value_5 = T.tensor_view(
                                        T.ptr_of(acc),
                                        layout=((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 4, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                                        shape=(64, 32),
                                    )
                                    lhs_ring_2 = T.tensor_view(
                                        ((2048 * (k // 16)) - (4096 * (k // 32))) + 2048,
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                        shape=(64, 16),
                                    )
                                    value_6 = T.tensor_view(
                                        T.ptr_of(lhs_ring_2),
                                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                        shape=(64, 16),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(value_6[0:0 + 64, 0:0 + 16]),
                                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                                        shape=(64, 16),
                                    )
                                    rhs_ring_2 = T.tensor_view(
                                        (1024 * (k // 16)) - (2048 * (k // 32)),
                                        dtype="bf16",
                                        storage="smem",
                                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                        shape=(16, 32),
                                    )
                                    value_7 = T.tensor_view(
                                        T.ptr_of(rhs_ring_2),
                                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                        shape=(16, 32),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(value_7[0:0 + 16, 0:0 + 32]),
                                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))) @ {B()},
                                        shape=(16, 32),
                                    )
                                    T.tiled_mma(
                                        value_5,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                    )
                T.fill(out, 0.0)
                acc_1 = T.alloc_tensor(
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
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    bias_r_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 32),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_8 = T.tensor_view(T.ptr_of(bias), layout=(64, 32), shape=(64, 32))
                    value_9 = T.tensor_view(
                        T.ptr_of(bias_r_dst),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    T.copy(value_8, value_9)
                    value_10 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 32),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                            "rmem",
                        ]
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 4, 4 @ scope_5.d2, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    T.binary(value_11, value_9, value_10, kind=BinaryKind.ADD)
                    T.relu(value_10, acc_1)
                    explicit_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((8 @ scope_5.d1, 2, 4 @ scope_5.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(explicit_dst),
                        layout=((8 @ scope_5.d1, 2, 4 @ scope_5.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(acc_1, value_12, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    automatic = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 2, 8 @ scope_5.d1, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                            "rmem",
                        ]
                    )
                    T.reduce(acc_1, automatic, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                    value_13 = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "f32",
                            ((4 @ scope_5.d0, 8 @ scope_5.d1, 2, 1), (16, 2, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.binary(value_12, automatic, value_13, kind=BinaryKind.ADD)
                    result = T.alloc_tensor(
                        tensor_type=Tensor[
                            (64, 1),
                            "bf16",
                            ((4 @ scope_5.d0, 8 @ scope_5.d1, 2, 1), (16, 2, 1, 0)),
                            "rmem",
                        ]
                    )
                    T.cast(value_13, result, dtype="bf16")
                    value_14 = T.tensor_view(T.ptr_of(out), layout=(64, 1), shape=(64, 1))
                    window = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 64, 0:0 + 1]), layout=(64, 1), shape=(64, 1)
                    )
                    T.copy(result, window)
