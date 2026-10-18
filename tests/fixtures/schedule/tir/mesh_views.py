# analysis target=nvidia.h200_sxm module=MeshViews function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r128B/w256B@logical,r128B/w256B@total,r128B/w256B@cta,r4B/w132B@thread;rmem:r928B/w640B@logical,r928B/w640B@total,r928B/w640B@cta,r60B/w20B@thread footprint=v12:26:128B;x:128B footprint-precision=exact peak=gmem:256B;rmem:256B persistent=gmem:128B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(4, 8), "f32"], out: Tensor[(8, 4), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (4, 8),
                "f32",
                ShardLayout(
                    layout=Layout((4, 8), (8, 1)),
                    attrs=(S(0), S(1)),
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")),
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
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")),
                ),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")) as scope:
            value_1 = T.tensor_view(T.ptr_of(x), layout=(4, 8), shape=(4, 8))
            value_2 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope.row, 8 @ scope.col), (8, 1)), shape=(4, 8)
            )
            dst_view = T.tensor_view(
                T.ptr_of(value_2[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope.row, 8 @ scope.col), (8, 1)),
                shape=(4, 8),
            )
            T.copy(value_1, dst_view)
            value_3 = T.tensor_view(
                T.ptr_of(value_2), layout=((4 @ scope.row, 8 @ scope.col), (8, 1)), shape=(4, 8)
            )
            value_4 = T.tensor_view(
                T.ptr_of(value_3), layout=((4 @ scope.row, 8 @ scope.col), (8, 1)), shape=(4, 8)
            )
            tile = T.tensor_view(
                T.ptr_of(value_4[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope.col, 4 @ scope.row), (1, 8)),
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
                    mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")),
                ),
                "rmem",
            ]
        )
        with Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")) as scope_1:
            mixed = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 8), "f32", ((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)), "rmem"
                ]
            )
            value_5 = T.tensor_view(
                T.ptr_of(value),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_6 = T.tensor_view(
                T.ptr_of(value_5),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_7 = T.tensor_view(
                T.ptr_of(value_6),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_8 = T.tensor_view(
                T.ptr_of(value_7),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_9 = T.tensor_view(
                T.ptr_of(value_8),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_10 = T.tensor_view(
                T.ptr_of(value_9),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            lhs_view = T.tensor_view(
                T.ptr_of(value_10[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_11 = T.tensor_view(
                T.ptr_of(value_6),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_12 = T.tensor_view(
                T.ptr_of(value_11),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_13 = T.tensor_view(
                T.ptr_of(value_12),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_14 = T.tensor_view(
                T.ptr_of(value_13),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_15 = T.tensor_view(
                T.ptr_of(value_14),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            rhs_view = T.tensor_view(
                T.ptr_of(value_15[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            T.binary(lhs_view, rhs_view, mixed, kind=BinaryKind.ADD)
            tile_1 = T.tensor_view(
                T.ptr_of(mixed[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope_1.col, 4 @ scope_1.row), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile_1, moved)
        with Mesh((Topology("thread", 32),), layout=(8, 4), names=("row", "col")) as scope_2:
            value_16 = T.tensor_view(T.ptr_of(out), layout=(8, 4), shape=(8, 4))
            T.fill(value_16, 0.0)
            result = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4),
                    "f32",
                    ShardLayout(
                        layout=Layout((8, 4), (4, 1)),
                        attrs=(S(1), S(0)),
                        mesh=Mesh((Topology("thread", 32),), layout=(4, 8), names=("row", "col")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                (Topology("thread", 32),), layout=(4, 8), names=("row", "col")
            ) as threads_5:
                value_17 = T.tensor_view(
                    T.ptr_of(transposed),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_17),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                value_19 = T.tensor_view(
                    T.ptr_of(value_18),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                value_20 = T.tensor_view(
                    T.ptr_of(value_19),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(moved),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((8 @ threads_5.col, 4 @ threads_5.row), (4, 1)),
                    shape=(8, 4),
                )
                T.binary(value_20, value_22, result, kind=BinaryKind.ADD)
            with Mesh(
                (Topology("thread", 32),), layout=(4, 8), names=("row", "col")
            ) as threads_6:
                value_23 = T.tensor_view(T.ptr_of(out), layout=(8, 4), shape=(8, 4))
                window = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 8, 0:0 + 4]), layout=(8, 4), shape=(8, 4)
                )
                T.copy(result, window)
