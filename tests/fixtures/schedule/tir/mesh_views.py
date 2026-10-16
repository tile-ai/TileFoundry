# analysis target=nvidia.h200_sxm module=MeshViews function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r128B/w256B@logical,r128B/w256B@total,r128B/w256B@cta,r4B/w132B@thread;rmem:r928B/w640B@logical,r928B/w640B@total,r928B/w640B@cta,r60B/w20B@thread footprint=v12:26:128B;x:128B footprint-precision=exact peak=gmem:256B;rmem:256B persistent=gmem:128B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


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
            T.copy(x, value)
            tile = T.tensor_view(
                T.ptr_of(value[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope.d0, 8 @ scope.d1), (8, 1)),
                shape=(4, 8),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope.d0, 8 @ scope.d1), (8, 1)),
                shape=(4, 8),
            )
            tile_2 = T.tensor_view(
                T.ptr_of(tile_1[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope.d1, 4 @ scope.d0), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile_2, transposed)
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
            tile_3 = T.tensor_view(
                T.ptr_of(value[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.d0, 8 @ scope_1.d1), (8, 1)),
                shape=(4, 8),
            )
            tile_4 = T.tensor_view(
                T.ptr_of(tile_3[0:0 + 4, 0:0 + 8]),
                layout=((4 @ scope_1.d0, 8 @ scope_1.d1), (8, 1)),
                shape=(4, 8),
            )
            T.binary(tile_3, tile_4, mixed, kind=BinaryKind.ADD)
            tile_5 = T.tensor_view(
                T.ptr_of(mixed[0:0 + 4, 0:0 + 8]),
                layout=((8 @ scope_1.d1, 4 @ scope_1.d0), (1, 8)),
                shape=(8, 4),
            )
            T.copy(tile_5, moved)
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
                T.binary(transposed, moved, result, kind=BinaryKind.ADD)
            with Mesh(
                (Topology("thread", 32),), layout=(4, 8), names=("d0", "d1")
            ) as threads_1:
                window = T.tensor_view(
                    T.ptr_of(out[0:0 + 8, 0:0 + 4]), layout=(8, 4), shape=(8, 4)
                )
                T.copy(result, window)
