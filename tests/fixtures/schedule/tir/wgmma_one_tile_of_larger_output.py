from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 32), "bf16"], out: Tensor[(128, 64), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        T.fill(out, 0.0)
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "bf16", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        with Mesh(
            (Topology("thread", 256),), Layout((2, 128), (128, 1)), names=("d0", "d1")
        ) as scope_3:
            lhs_stages = (T.tensor_view(1024, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)), T.tensor_view(2048, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)))
            rhs_stages = (T.tensor_view(0, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)), T.tensor_view(512, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)))
            with Mesh(
                (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
            ) as threads:
                T.fill(acc, 0.0)
            for k in range(0, 32, 16):
                with scope_3[:1, :32] as scope:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 64, k:k + 16]),
                        layout=Layout((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 32]),
                        layout=Layout((16, 32), (32, 1)),
                        shape=(16, 32),
                    )
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 16) % 2])
                with scope_3[1:] as scope_1:
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                    ) as threads_3:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 16, 16):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 32]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 4), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 32),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 16) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 16) % 2][o_k:o_k + 16, o_n:o_n + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 32),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                    )
            with scope_3[1:] as scope_2:
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_4:
                    value_view = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.cast(acc, value_view)
                    value_view_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[0:0 + 64, 0:0 + 32]),
                        layout=Layout((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    T.copy(value_view_1, window)
