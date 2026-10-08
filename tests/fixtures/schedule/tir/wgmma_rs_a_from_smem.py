# analysis target=nvidia.h200_sxm module=WGMMA_RS_A_FROM_SMEM function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w0@logical,r6.00KB/w0@total,r6.00KB/w0@cta,r6.00KB/w0@thread;rmem:r28.06KB/w32.00KB@logical,r28.06KB/w32.00KB@total,r28.06KB/w32.00KB@cta,r288B/w256B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r160B/w6.00KB@thread footprint=a:2.00KB;b:1.00KB footprint-precision=exact peak=gmem:6.00KB;rmem:10.00KB;smem:6.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 64), "fp8e4m3"], b: Tensor[(64, 32), "fp8e4m3", Layout((64, 32), (1, 64))], out: Tensor[(64, 32), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        frag = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32),
                "fp8e4m3",
                Layout((8, 2, 4, 4, 4, 2), (1, 8, 16, 64, 256, 1024)),
                "rmem",
            ]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "bf16", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
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
            lhs_stages = (T.tensor_view(2048, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)), T.tensor_view(4096, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)))
            rhs_stages = (T.tensor_view(0, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)), T.tensor_view(1024, dtype='fp8e4m3', storage=StorageKind.SMEM, layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)))
            for k in range(0, 64, 32):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 64, k:k + 32]),
                        layout=Layout((64, 32), (64, 1)),
                        shape=(64, 32),
                    )
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 32) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 32, 0:0 + 32]),
                        layout=Layout((32, 32), (1, 64)),
                        shape=(32, 32),
                    )
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                    ) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 32) % 2])
                with scope[1:] as scope_2:
                    with Mesh(
                        (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                    ) as threads_3:
                        dst_frame = T.tensor_view(
                            T.ptr_of(frag[0:0 + 64, 0:0 + 32]),
                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 4, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 256, 1024)),
                            shape=(64, 32),
                        )
                        T.copy(lhs_stages[(k // 32) % 2], dst_frame)
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 32, 32):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 32]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 4), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 32),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(frag[o_m:o_m + 64, o_k:o_k + 32]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 4, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 256, 1024)),
                                        shape=(64, 32),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 32) % 2][o_k:o_k + 32, o_n:o_n + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(32, 32),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, dtype='fp8e4m3', form=T.cuda.sm90.Form.RS, mesh=threads_3),
                                    )
            with scope[1:] as scope_3:
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_5:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.cast(src_frame, dst_frame_1, dtype='bf16')
        with Mesh(
            (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
        ) as threads_6:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                shape=(64, 32),
            )
            T.copy(result_view, out)
