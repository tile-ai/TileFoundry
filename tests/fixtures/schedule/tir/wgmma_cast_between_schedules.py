# analysis target=nvidia.h200_sxm module=WGMMA_CAST_BETWEEN_SCHEDULES function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16.00KB/w0@logical,r16.00KB/w0@total,r14.00KB/w0@cta,r4.19KB/w0@thread;rmem:r62.81KB/w54.88KB@logical,r62.81KB/w54.88KB@total,r59.81KB/w51.88KB@cta,r728B/w604B@thread;smem:r320B/w320B@logical,r320B/w320B@total,r320B/w320B@cta,r72B/w260B@thread footprint=a:2.00KB;b_f32:2.00KB;bias:8.00KB footprint-precision=lower_bound peak=gmem:16.00KB;rmem:11.00KB;smem:6.00KB persistent=gmem:16.00KB

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b_f32: Tensor[(32, 32), "f32"], bias: Tensor[(64, 32), "f32"], out: Tensor[(64, 1), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        b = T.alloc_tensor(
            tensor_type=Tensor[(16, 32), "bf16", Layout((32, 16), (16, 1)), "rmem"]
        )
        b_tile = T.alloc_tensor(
            tensor_type=Tensor[(16, 32), "f32", Layout((32, 16), (16, 1)), "rmem"]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "bf16", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "f32", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        explicit = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "f32", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        acc_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        bias_r = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        automatic = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "f32", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        with Mesh(
            (Topology("thread", 256),), Layout((2, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            with Mesh(
                (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
            ) as threads:
                T.fill(acc, 0.0)
            lhs_stages = (T.tensor_view(2048, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)), T.tensor_view(4096, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)))
            rhs_stages = (T.tensor_view(0, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)), T.tensor_view(1024, dtype='bf16', storage=StorageKind.SMEM, layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)))
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_1:
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
                        T.ptr_of(b_f32[k:k + 16, 0:0 + 32]),
                        layout=Layout((16, 32), (32, 1)),
                        shape=(16, 32),
                    )
                    with Mesh(
                        (Topology("thread", 256),), Layout((32,), (1,)), names=("d0",)
                    ) as threads_2:
                        dst_frame = T.tensor_view(
                            T.ptr_of(b_tile[0:0 + 16, 0:0 + 32]),
                            layout=((32 @ threads_2.d0, 16), (16, 1)),
                            shape=(16, 32),
                        )
                        T.copy(tile_1, dst_frame)
                        src_frame = T.tensor_view(
                            T.ptr_of(b_tile[0:0 + 16, 0:0 + 32]),
                            layout=((32 @ threads_2.d0, 16), (16, 1)),
                            shape=(16, 32),
                        )
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(b[0:0 + 16, 0:0 + 32]),
                            layout=((32 @ threads_2.d0, 16), (16, 1)),
                            shape=(16, 32),
                        )
                        T.cast(src_frame, dst_frame_1, dtype='bf16')
                        src_frame_1 = T.tensor_view(
                            T.ptr_of(b[0:0 + 16, 0:0 + 32]),
                            layout=((32 @ threads_2.d0, 16), (16, 1)),
                            shape=(16, 32),
                        )
                        T.copy(src_frame_1, rhs_stages[(k // 16) % 2])
                with scope[1:] as scope_3:
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                    ) as threads_5:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 16, 16):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 32]),
                                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 4), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 32),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 16) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 16) % 2][o_k:o_k + 16, o_n:o_n + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_5,
                                        ),
                                        shape=(16, 32),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, mesh=threads_5),
                                    )
            with scope[1:] as scope_4:
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_6:
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(bias_r[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.copy(bias, dst_frame_2)
                    lhs_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(bias_r[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.binary(lhs_frame, rhs_frame, dst_frame_3, kind=BinaryKind.ADD)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.relu(src_frame_2, dst_frame_4)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(explicit[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_3,
                        dst_frame_5,
                        axes=(1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(automatic[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_4,
                        dst_frame_6,
                        axes=(1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(explicit[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(automatic[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.binary(lhs_frame_1, rhs_frame_1, dst_frame_7, kind=BinaryKind.ADD)
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(result[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.cast(src_frame_5, dst_frame_8, dtype='bf16')
        with Mesh(
            (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
        ) as threads_13:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 64, 0:0 + 1]),
                layout=((8 @ threads_13.d1, 2, 4 @ threads_13.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                shape=(64, 1),
            )
            T.copy(result_view, out)
