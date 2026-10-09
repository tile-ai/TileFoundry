# analysis target=nvidia.h200_sxm module=FP8_BLOCK_SCALED_GEMM function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r130.02KB/w0@logical,r130.02KB/w0@total,r130.02KB/w0@cta,r128.05KB/w0@thread;rmem:r1.31MB/w1.16MB@logical,r1.31MB/w1.35MB@total,r1.31MB/w1.35MB@cta,r5.55KB/w5.42KB@thread;smem:r128.00KB/w128.00KB@logical,r128.00KB/w128.00KB@total,r128.00KB/w128.00KB@cta,r17.00KB/w128.00KB@thread footprint=a:16.00KB;a_scale:512B;b:16.00KB;b_scale:4B footprint-precision=exact peak=gmem:130.02KB;rmem:64.50KB;smem:64.00KB persistent=gmem:130.02KB

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Swizzle, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 512), "fp8e4m3"], b: Tensor[(512, 128), "fp8e4m3", Layout((512, 128), (1, 512))], a_scale: Tensor[(128, 4), "f32"], b_scale: Tensor[(4, 1), "f32"], out: Tensor[(128, 128), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        acc_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        part = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "f32",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        row_scale = T.alloc_tensor(
            tensor_type=Tensor[(128, 1), "f32", Layout((2, 8, 2, 4), (64, 1, 8, 16)), "rmem"]
        )
        tile_scale = T.alloc_tensor(
            tensor_type=Tensor[(1, 1), "f32", Layout((1, 1), (1, 1)), "rmem"]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (128, 128),
                "bf16",
                Layout((2, 8, 2, 4, 2, 4, 16), (8192, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        with Mesh(
            (Topology("thread", 384),), Layout((3, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            with Mesh(
                (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_stages = (T.tensor_view(32768, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=ComposedLayout(
                    inner=Swizzle(3, 4, 3),
                    offset=0,
                    outer=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))),
                ), shape=(128, 128)), T.tensor_view(49152, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=ComposedLayout(
                    inner=Swizzle(3, 4, 3),
                    offset=0,
                    outer=Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))),
                ), shape=(128, 128)))
            rhs_stages = (T.tensor_view(0, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=ComposedLayout(
                    inner=Swizzle(3, 4, 3),
                    offset=0,
                    outer=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))),
                ), shape=(128, 128)), T.tensor_view(16384, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=ComposedLayout(
                    inner=Swizzle(3, 4, 3),
                    offset=0,
                    outer=Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))),
                ), shape=(128, 128)))
            for kb in range(0, 4, 1):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 128, kb * 128:kb * 128 + 128]),
                        layout=Layout((128, 128), (512, 1)),
                        shape=(128, 128),
                    )
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[kb % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[kb * 128:kb * 128 + 128, 0:0 + 128]),
                        layout=Layout((128, 128), (1, 512)),
                        shape=(128, 128),
                    )
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[kb % 2])
                with scope[1:] as scope_2:
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                    ) as threads_3:
                        T.fill(part, 0.0)
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                    ) as threads_4:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 128, 128):
                                for o_k in range(0, 128, 128):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(part[o_m:o_m + 64, o_n:o_n + 128]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m:o_m + 64, o_k:o_k + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k:o_k + 32, o_n:o_n + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(part[o_m:o_m + 64, o_n:o_n + 128]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m:o_m + 64, o_k + 32:o_k + 32 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k + 32:o_k + 32 + 32, o_n:o_n + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_2 = T.tensor_view(
                                        T.ptr_of(part[o_m:o_m + 64, o_n:o_n + 128]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_2 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m:o_m + 64, o_k + 64:o_k + 64 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=64,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_2 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k + 64:o_k + 64 + 32, o_n:o_n + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=64,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_2,
                                        lhs_view_2,
                                        rhs_view_2,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                                    acc_view_3 = T.tensor_view(
                                        T.ptr_of(part[o_m:o_m + 64, o_n:o_n + 128]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_3 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m:o_m + 64, o_k + 96:o_k + 96 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=96,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_3 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k + 96:o_k + 96 + 32, o_n:o_n + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=96,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_3,
                                        lhs_view_3,
                                        rhs_view_3,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=256,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                    ) as threads_5:
                        for o_m_1 in range(64, 128, 64):
                            for o_n_1 in range(0, 128, 128):
                                for o_k_1 in range(0, 128, 128):
                                    acc_view_4 = T.tensor_view(
                                        T.ptr_of(part[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 128]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_4 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_4 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k_1:o_k_1 + 32, o_n_1:o_n_1 + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=0,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_4,
                                        lhs_view_4,
                                        rhs_view_4,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_5 = T.tensor_view(
                                        T.ptr_of(part[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 128]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_5 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m_1:o_m_1 + 64, o_k_1 + 32:o_k_1 + 32 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_5 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k_1 + 32:o_k_1 + 32 + 32, o_n_1:o_n_1 + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=32,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_5,
                                        lhs_view_5,
                                        rhs_view_5,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_6 = T.tensor_view(
                                        T.ptr_of(part[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 128]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_6 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m_1:o_m_1 + 64, o_k_1 + 64:o_k_1 + 64 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=64,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_6 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k_1 + 64:o_k_1 + 64 + 32, o_n_1:o_n_1 + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=64,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_6,
                                        lhs_view_6,
                                        rhs_view_6,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                                    acc_view_7 = T.tensor_view(
                                        T.ptr_of(part[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 128]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 16), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 128),
                                    )
                                    lhs_view_7 = T.tensor_view(
                                        T.ptr_of(lhs_stages[kb % 2][o_m_1:o_m_1 + 64, o_k_1 + 96:o_k_1 + 96 + 32]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=96,
                                                outer=Layout(((8, 8), 32), ((1024, 128), 1)),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view_7 = T.tensor_view(
                                        T.ptr_of(rhs_stages[kb % 2][o_k_1 + 96:o_k_1 + 96 + 32, o_n_1:o_n_1 + 128]),
                                        layout=ShardLayout(
                                            layout=ComposedLayout(
                                                inner=Swizzle(3, 4, 3),
                                                offset=96,
                                                outer=Layout((32, (16, 8)), (1, (1024, 128))),
                                            ),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(32, 128),
                                    )
                                    T.tiled_mma(
                                        acc_view_7,
                                        lhs_view_7,
                                        rhs_view_7,
                                        atom=T.cuda.sm90.Wgmma(n=128, dtype='fp8e4m3', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
                    tile_2 = T.tensor_view(
                        T.ptr_of(a_scale[0:0 + 128, kb:kb + 1]),
                        layout=Layout((128, 1), (4, 1)),
                        shape=(128, 1),
                    )
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                    ) as threads_6:
                        dst_frame = T.tensor_view(
                            T.ptr_of(row_scale[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_6.d0, 8 @ threads_6.d2, 2, 4 @ threads_6.d1), (64, 1, 8, 16)),
                            shape=(128, 1),
                        )
                        T.copy(tile_2, dst_frame)
                        lhs_frame = T.tensor_view(
                            T.ptr_of(part[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_6.d0, 8 @ threads_6.d2, 2, 4 @ threads_6.d1, 2, 4 @ threads_6.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(row_scale[0:0 + 128, 0:0 + 1]),
                            layout=((2 @ threads_6.d0, 8 @ threads_6.d2, 2, 4 @ threads_6.d1), (64, 1, 8, 16)),
                            shape=(128, 1),
                        )
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_6.d0, 8 @ threads_6.d2, 2, 4 @ threads_6.d1, 2, 4 @ threads_6.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        T.binary(lhs_frame, rhs_frame, dst_frame_1, kind=BinaryKind.MUL)
                    tile_3 = T.tensor_view(
                        T.ptr_of(b_scale[kb:kb + 1, 0:0 + 1]),
                        layout=Layout((1, 1), (1, 1)),
                        shape=(1, 1),
                    )
                    with Mesh(
                        (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                    ) as threads_8:
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(tile_scale[0:0 + 1, 0:0 + 1]),
                            layout=((1, 1), (1, 1), {threads_8.d0 @ B()}),
                            shape=(1, 1),
                        )
                        T.copy(tile_3, dst_frame_2)
                        lhs_frame_1 = T.tensor_view(
                            T.ptr_of(value_1[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_8.d0, 8 @ threads_8.d2, 2, 4 @ threads_8.d1, 2, 4 @ threads_8.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        rhs_frame_1 = T.tensor_view(
                            T.ptr_of(tile_scale[0:0 + 1, 0:0 + 1]),
                            layout=((1, 1), (1, 1), {threads_8.d0 @ B()}),
                            shape=(1, 1),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(value[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_8.d0, 8 @ threads_8.d2, 2, 4 @ threads_8.d1, 2, 4 @ threads_8.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        T.binary(lhs_frame_1, rhs_frame_1, dst_frame_3, kind=BinaryKind.MUL)
                        lhs_frame_2 = T.tensor_view(
                            T.ptr_of(acc[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_8.d0, 8 @ threads_8.d2, 2, 4 @ threads_8.d1, 2, 4 @ threads_8.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        rhs_frame_2 = T.tensor_view(
                            T.ptr_of(value[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_8.d0, 8 @ threads_8.d2, 2, 4 @ threads_8.d1, 2, 4 @ threads_8.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        dst_frame_4 = T.tensor_view(
                            T.ptr_of(acc_1[0:0 + 128, 0:0 + 128]),
                            layout=((2 @ threads_8.d0, 8 @ threads_8.d2, 2, 4 @ threads_8.d1, 2, 4 @ threads_8.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                            shape=(128, 128),
                        )
                        T.binary(lhs_frame_2, rhs_frame_2, dst_frame_4, kind=BinaryKind.ADD)
                with Mesh(
                    (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                ) as threads_11:
                    acc_1_view = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ threads_11.d0, 8 @ threads_11.d2, 2, 4 @ threads_11.d1, 2, 4 @ threads_11.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 128),
                    )
                    T.copy(acc_1_view, acc)
            with scope[1:] as scope_3:
                with Mesh(
                    (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
                ) as threads_12:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ threads_12.d0, 8 @ threads_12.d2, 2, 4 @ threads_12.d1, 2, 4 @ threads_12.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 128),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(result[0:0 + 128, 0:0 + 128]),
                        layout=((2 @ threads_12.d0, 8 @ threads_12.d2, 2, 4 @ threads_12.d1, 2, 4 @ threads_12.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                        shape=(128, 128),
                    )
                    T.cast(src_frame, dst_frame_5, dtype='bf16')
        with Mesh(
            (Topology("thread", 384),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
), names=("d0", "d1", "d2", "d3")
        ) as threads_13:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 128, 0:0 + 128]),
                layout=((2 @ threads_13.d0, 8 @ threads_13.d2, 2, 4 @ threads_13.d1, 2, 4 @ threads_13.d3, 16), (8192, 1, 8, 16, 64, 128, 512)),
                shape=(128, 128),
            )
            T.copy(result_view, out)
