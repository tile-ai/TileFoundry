from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(4, 8), "f32"], out: Tensor[(8, 4), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (4, 8),
                "f32",
                ShardLayout(
                    layout=Layout((4, 8), (8, 1)),
                    attrs=(S(0), S(1)),
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        transposed = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 4),
                "f32",
                ShardLayout(
                    layout=Layout((8, 4), (4, 1)),
                    attrs=(S(1), S(0)),
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")) as scope:
            value_1 = T.tensor_view(T.ptr_of(x), layout=(4, 8), shape=(4, 8))
            value_2 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope.d0, 8 @ scope.d1), (8, 1)), shape=(4, 8)
            )
            T.copy(value_1, value_2)
            value_3 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope.d0, 8 @ scope.d1), (8, 1)), shape=(4, 8)
            )
            tile = T.tensor_view(
                T.ptr_of(value_3[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope.d1, 4 @ scope.d0), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile, transposed)
        moved = T.alloc_tensor(
            tensor_type=Tensor[
                (8, 4),
                "f32",
                ShardLayout(
                    layout=Layout((8, 4), (4, 1)),
                    attrs=(S(1), S(0)),
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")),
                ),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")) as scope_1:
            mixed = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 8), "f32", ((4 @ scope_1.d0, 8 @ scope_1.d1), (8, 1)), "rmem"
                ]
            )
            value_4 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope_1.d0, 8 @ scope_1.d1), (8, 1)), shape=(4, 8)
            )
            value_5 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope_1.d0, 8 @ scope_1.d1), (8, 1)), shape=(4, 8)
            )
            T.binary(value_4, value_5, mixed, kind=BinaryKind.ADD)
            tile_1 = T.tensor_view(
                T.ptr_of(mixed[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope_1.d1, 4 @ scope_1.d0), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile_1, moved)
        with Mesh((Topology("thread", 32),), layout=(8, 4), names=("d0", "d1")) as scope_2:
            T.fill(out, 0.0)
            result = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 4), (4, 1)),
                        attrs=(S(1), S(0)),
                        mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")),
                    ),
                    "rmem",
                ]
            )
            with Mesh((Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")) as threads:
                value_6 = T.tensor_view(
                    T.ptr_of(transposed),
                    layout=((8 @ threads.d1, 4 @ threads.d0), (4, 1)),
                    shape=(8, 4),
                )
                value_7 = T.tensor_view(
                    T.ptr_of(moved),
                    layout=((8 @ threads.d1, 4 @ threads.d0), (4, 1)),
                    shape=(8, 4),
                )
                T.binary(value_6, value_7, result, kind=BinaryKind.ADD)
            with Mesh(
                (Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")
            ) as threads_1:
                value_8 = T.tensor_view(T.ptr_of(out), layout=(8, 4), shape=(8, 4))
                window = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 8, 0:0 + 4]), layout=(8, 4), shape=(8, 4)
                )
                T.copy(result, window)
