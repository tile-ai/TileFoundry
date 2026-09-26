from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Swizzle, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 256),
                "bf16",
                Layout((2, 8, 2, 4, 2, 4, 32), (16384, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        with Mesh(
            (Topology("thread", 384),), Layout((3, 128), (128, 1)), names=("d0", "d1")
        ) as scope_4:
            for m in range(0, 8192, 128):
                for n in range(0, 17408, 256):
                    lhs_stages = (T.tensor_view(65536, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))),
                        ), shape=(128, 64)), T.tensor_view(73728, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))),
                        ), shape=(128, 64)), T.tensor_view(81920, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))),
                        ), shape=(128, 64)), T.tensor_view(90112, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))),
                        ), shape=(128, 64)))
                    rhs_stages = (T.tensor_view(0, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))),
                        ), shape=(64, 256)), T.tensor_view(16384, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))),
                        ), shape=(64, 256)), T.tensor_view(32768, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))),
                        ), shape=(64, 256)), T.tensor_view(49152, dtype='bf16', storage=StorageKind.SMEM, layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))),
                        ), shape=(64, 256)))
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                    ) as threads:
                        T.fill(acc, 0.0)
                    for k in range(0, 5120, 64):
                        with scope_4[:1, :32] as scope:
                            tile = T.tensor_view(
                                T.ptr_of(a[m:m + 128, k:k + 64]),
                                layout=Layout((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            with Mesh(
                                (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                            ) as threads_1:
                                T.copy_async_tensor(tile, lhs_stages[(k // 64) % 4])
                            tile_1 = T.tensor_view(
                                T.ptr_of(b[k:k + 64, n:n + 256]),
                                layout=Layout((64, 256), (17408, 1)),
                                shape=(64, 256),
                            )
                            with Mesh(
                                (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                            ) as threads_2:
                                T.copy_async_tensor(tile_1, rhs_stages[(k // 64) % 4])
                        with scope_4[1:] as scope_1:
                            with Mesh(
                                (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                            ) as threads_3:
                                for o_m in range(0, 64, 64):
                                    for o_n in range(0, 256, 256):
                                        for o_k in range(0, 64, 64):
                                            acc_view = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m:o_m + 64, o_k:o_k + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k:o_k + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view,
                                                lhs_view,
                                                rhs_view,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                            )
                                            acc_view_1 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_1 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=16,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_1 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_1,
                                                lhs_view_1,
                                                rhs_view_1,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                            )
                                            acc_view_2 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_2 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=32,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_2 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_2,
                                                lhs_view_2,
                                                rhs_view_2,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                            )
                                            acc_view_3 = T.tensor_view(
                                                T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 256]),
                                                layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_3 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=48,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_3 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_3,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_3,
                                                lhs_view_3,
                                                rhs_view_3,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_3),
                                            )
                            with Mesh(
                                (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=256,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                            ) as threads_4:
                                for o_m_1 in range(64, 128, 64):
                                    for o_n_1 in range(0, 256, 256):
                                        for o_k_1 in range(0, 64, 64):
                                            acc_view_4 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_4 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_4 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_4,
                                                lhs_view_4,
                                                rhs_view_4,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                            )
                                            acc_view_5 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_5 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 16:o_k_1 + 16 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=16,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_5 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k_1 + 16:o_k_1 + 16 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_5,
                                                lhs_view_5,
                                                rhs_view_5,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                            )
                                            acc_view_6 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_6 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=32,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_6 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k_1 + 32:o_k_1 + 32 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_6,
                                                lhs_view_6,
                                                rhs_view_6,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                            )
                                            acc_view_7 = T.tensor_view(
                                                T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 256]),
                                                layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 32), (1, 8, 16, 64, 128, 512)),
                                                shape=(64, 256),
                                            )
                                            lhs_view_7 = T.tensor_view(
                                                T.ptr_of(lhs_stages[(k // 64) % 4][o_m_1:o_m_1 + 64, o_k_1 + 48:o_k_1 + 48 + 16]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=48,
                                                        outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(64, 16),
                                            )
                                            rhs_view_7 = T.tensor_view(
                                                T.ptr_of(rhs_stages[(k // 64) % 4][o_k_1 + 48:o_k_1 + 48 + 16, o_n_1:o_n_1 + 256]),
                                                layout=ShardLayout(
                                                    layout=ComposedLayout(
                                                        inner=Swizzle(3, 4, 3),
                                                        offset=0,
                                                        outer=Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))),
                                                    ),
                                                    attrs=(B(), B(), B()),
                                                    mesh=threads_4,
                                                ),
                                                shape=(16, 256),
                                            )
                                            T.tiled_mma(
                                                acc_view_7,
                                                lhs_view_7,
                                                rhs_view_7,
                                                atom=T.cuda.sm90.Wgmma(n=256, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                            )
                    copy = T.tensor_view(
                        0,
                        dtype='bf16',
                        storage=StorageKind.SMEM,
                        layout=ComposedLayout(
                            inner=Swizzle(3, 4, 3),
                            offset=0,
                            outer=Layout((128, (4, 64)), (64, (8192, 1))),
                        ),
                        shape=(128, 256),
                    )
                    with scope_4[1:] as scope_2:
                        with Mesh(
                            (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                        ) as threads_5:
                            value_view = T.tensor_view(
                                T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.cast(acc, value_view)
                            src_frame = T.tensor_view(
                                T.ptr_of(value[0:0 + 128, 0:0 + 256]),
                                layout=((2 @ threads_5.d0, 8 @ threads_5.d2, 2, 4 @ threads_5.d1, 2, 4 @ threads_5.d3, 32), (16384, 1, 8, 16, 64, 128, 512)),
                                shape=(128, 256),
                            )
                            T.copy(src_frame, copy)
                    with scope_4[:1, :32] as scope_3:
                        with Mesh(
                            (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                        ) as threads_7:
                            window = T.tensor_view(
                                T.ptr_of(out[m:m + 128, n:n + 256]),
                                layout=Layout((128, 256), (17408, 1)),
                                shape=(128, 256),
                            )
                            T.copy_async_tensor(copy, window)
