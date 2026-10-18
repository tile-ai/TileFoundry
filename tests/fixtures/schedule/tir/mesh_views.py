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
            dst_view = T.tensor_view(
                T.ptr_of(value[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope.row, 8 @ scope.col), (8, 1)),
                shape=(4, 8),
            )
            T.copy(x, dst_view)
            value_1 = T.tensor_view(
                T.ptr_of(value), layout=((4 @ scope.row, 8 @ scope.col), (8, 1)), shape=(4, 8)
            )
            value_2 = T.tensor_view(
                T.ptr_of(value_1), layout=((4 @ scope.row, 8 @ scope.col), (8, 1)), shape=(4, 8)
            )
            tile = T.tensor_view(
                T.ptr_of(value_2), layout=((8 @ scope.col, 4 @ scope.row), (1, 8)), shape=(8, 4)
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
            value_3 = T.tensor_view(
                T.ptr_of(value),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            lhs_view = T.tensor_view(
                T.ptr_of(value_3[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            value_4 = T.tensor_view(
                T.ptr_of(value_3),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            rhs_view = T.tensor_view(
                T.ptr_of(value_4[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.row, 8 @ scope_1.col), (8, 1)),
                shape=(4, 8),
            )
            T.binary(lhs_view, rhs_view, mixed, kind=BinaryKind.ADD)
            tile_1 = T.tensor_view(
                T.ptr_of(mixed),
                layout=((8 @ scope_1.col, 4 @ scope_1.row), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile_1, moved)
        with Mesh((Topology("thread", 32),), layout=(8, 4), names=("row", "col")) as scope_2:
            T.fill(out, 0.0)
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
                T.binary(transposed, moved, result, kind=BinaryKind.ADD)
            with Mesh(
                (Topology("thread", 32),), layout=(4, 8), names=("row", "col")
            ) as threads_6:
                T.copy(result, out)
