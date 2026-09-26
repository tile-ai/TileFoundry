from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(a: Tensor[(16, 32), "bf16"], b: Tensor[(32, 8), "bf16"], out: Tensor[(16, 8), "bf16"]):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[(16, 8), "f32", Layout((2, 4, 8, 2), (1, 2, 8, 64)), "rmem"]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[(16, 8), "bf16", Layout((2, 4, 8, 2), (1, 2, 8, 64)), "rmem"]
        )
        with Mesh(
            (Topology("thread", 64),), Layout((2, 32), (32, 1)), names=("d0", "d1")
        ) as scope_3:
            lhs_stages = (T.tensor_view(256, dtype='bf16', storage=StorageKind.SMEM, layout=Layout((16, 16), (16, 1)), shape=(16, 16)), T.tensor_view(512, dtype='bf16', storage=StorageKind.SMEM, layout=Layout((16, 16), (16, 1)), shape=(16, 16)))
            rhs_stages = (T.tensor_view(0, dtype='bf16', storage=StorageKind.SMEM, layout=Layout((16, 8), (8, 1)), shape=(16, 8)), T.tensor_view(128, dtype='bf16', storage=StorageKind.SMEM, layout=Layout((16, 8), (8, 1)), shape=(16, 8)))
            with Mesh(
                (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=32,
    outer=Layout((4, 8), (1, 4)),
), names=("d0", "d1")
            ) as threads:
                T.fill(acc, 0.0)
            for k in range(0, 32, 16):
                with scope_3[:1] as scope:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 16, k:k + 16]),
                        layout=Layout((16, 16), (32, 1)),
                        shape=(16, 16),
                    )
                    with Mesh(
                        (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 8]),
                        layout=Layout((16, 8), (8, 1)),
                        shape=(16, 8),
                    )
                    with Mesh(
                        (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 16) % 2])
                with scope_3[1:] as scope_1:
                    ldmatrix = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 16),
                            "bf16",
                            Layout((2, 4, 2, 8, 2), (1, 2, 8, 16, 128)),
                            "rmem",
                        ]
                    )
                    with Mesh(
                        (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=32,
    outer=Layout((4, 8), (1, 4)),
), names=("d0", "d1")
                    ) as threads_3:
                        T.ldmatrix(lhs_stages[(k // 16) % 2], ldmatrix)
                    copy = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 8), "bf16", Layout((8, 2, 4, 2), (1, 8, 16, 64)), "rmem"
                        ]
                    )
                    with Mesh(
                        (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=32,
    outer=Layout((4, 8), (1, 4)),
), names=("d0", "d1")
                    ) as threads_4:
                        T.copy(rhs_stages[(k // 16) % 2], copy)
                        for o_m in range(0, 16, 16):
                            for o_n in range(0, 8, 8):
                                for o_k in range(0, 16, 16):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 16, o_n:o_n + 8]),
                                        layout=((2, 4 @ threads_4.d0, 8 @ threads_4.d1, 2), (1, 2, 8, 64)),
                                        shape=(16, 8),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(ldmatrix[o_m:o_m + 16, o_k:o_k + 16]),
                                        layout=((2, 4 @ threads_4.d0, 2, 8 @ threads_4.d1, 2), (1, 2, 8, 16, 128)),
                                        shape=(16, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(copy[o_k:o_k + 16, o_n:o_n + 8]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2), (1, 8, 16, 64)),
                                        shape=(16, 8),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm80.Mma(mesh=threads_4),
                                    )
            with scope_3[1:] as scope_2:
                with Mesh(
                    (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=32,
    outer=Layout((4, 8), (1, 4)),
), names=("d0", "d1")
                ) as threads_6:
                    value_view = T.tensor_view(
                        T.ptr_of(value[0:0 + 16, 0:0 + 8]),
                        layout=((2, 4 @ threads_6.d0, 8 @ threads_6.d1, 2), (1, 2, 8, 64)),
                        shape=(16, 8),
                    )
                    T.cast(acc, value_view)
        with Mesh(
            (Topology("thread", 64),), ComposedLayout(
    inner=None,
    offset=32,
    outer=Layout((4, 8), (1, 4)),
), names=("d0", "d1")
        ) as threads_7:
            value_view_1 = T.tensor_view(
                T.ptr_of(value[0:0 + 16, 0:0 + 8]),
                layout=((2, 4 @ threads_7.d0, 8 @ threads_7.d1, 2), (1, 2, 8, 64)),
                shape=(16, 8),
            )
            T.copy(value_view_1, out)
