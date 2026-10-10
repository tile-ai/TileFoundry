# analysis target=nvidia.h200_sxm module=SparseDecode function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r17.00KB/w0@logical,r17.00KB/w0@total,r17.00KB/w0@cta,r17.00KB/w0@thread;rmem:r181.58KB/w153.52KB@logical,r181.58KB/w153.52KB@total,r181.58KB/w153.52KB@cta,r6.29KB/w2.11KB@thread;smem:r40.00KB/w48.00KB@logical,r40.00KB/w48.00KB@total,r40.00KB/w48.00KB@cta,r18.38KB/w48.00KB@thread footprint=idx:512B;kv:32.00KB;q:128B footprint-precision=exact peak=gmem:64.50KB;rmem:32.25KB;smem:16.00KB persistent=gmem:64.50KB
#   buffer=kv holds=32.13KB time=r space=none reuse=1.97MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.types import B, ComposedLayout, Layout, Mesh, ShardLayout, Swizzle, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    q: Tensor[(1, 2, 128, 64), "bf16"], kv: Tensor[(1, 256, 1, 64), "bf16"], idx: Tensor[(64,), "i64"], out: Tensor[(64, 64), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        tile = T.tensor_view(
            T.ptr_of(q[0:0 + 1, 1:1 + 1, 32:32 + 64, 0:0 + 64]),
            layout=Layout((1, 1, 64, 64), (16384, 8192, 64, 1)),
            shape=(1, 1, 64, 64),
        )
        tile_1 = T.tensor_view(
            T.ptr_of(tile[0:0 + 1, 0:0 + 1, 0:0 + 64, 0:0 + 64]),
            layout=Layout((64, 64), (64, 1)),
            shape=(64, 64),
        )
        tile_2 = T.tensor_view(
            T.ptr_of(kv[0:0 + 1, 0:0 + 256, 0:0 + 1, 0:0 + 64]),
            layout=Layout((256, 64), (64, 1)),
            shape=(256, 64),
        )
        qs = T.tensor_view(
            0,
            dtype='bf16',
            storage=StorageKind.SMEM,
            layout=ComposedLayout(
                inner=Swizzle(3, 4, 3),
                offset=0,
                outer=Layout((64, 64), (64, 1)),
            ),
            shape=(64, 64),
        )
        ks = T.tensor_view(
            8192,
            dtype='bf16',
            storage=StorageKind.SMEM,
            layout=ComposedLayout(
                inner=Swizzle(3, 4, 3),
                offset=0,
                outer=Layout((64, 64), (64, 1)),
            ),
            shape=(64, 64),
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", Layout((2, 4, 8, 2, 4, 8), (8, 16, 1, 64, 128, 512)), "rmem"
            ]
        )
        o = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        pr = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64),
                "bf16",
                Layout((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
                "rmem",
            ]
        )
        p = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "bf16", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        p_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        s = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        live = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", Layout((64,), (1,)), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", Layout((64,), (1,)), "rmem"])
        ii = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", Layout((64,), (1,)), "rmem"])
        ii_1 = T.alloc_tensor(tensor_type=Tensor[(64,), "i64", Layout((64,), (1,)), "rmem"])
        value_3 = T.alloc_tensor(tensor_type=Tensor[(64,), "bool", Layout((64,), (1,)), "rmem"])
        s_1 = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 64), "f32", Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        peak = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "f32", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        denom = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 1), "f32", Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)), "rmem"
            ]
        )
        with Mesh(
            (Topology("thread", 256),), Layout((2, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            with scope[:1, :32] as scope_1:
                T.fill(qs, 0.0)
                T.fill(ks, 0.0)
                for r in range(0, 64, 1):
                    tile_3 = T.tensor_view(
                        T.ptr_of(idx[r:r + 1]), layout=Layout((1,), (1,)), shape=(1,)
                    )
                    with scope_1 as scope_2:
                        with Mesh(
                            (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                        ) as threads:
                            window = T.tensor_view(
                                T.ptr_of(ks[r:r + 1, 0:0 + 64]),
                                layout=ComposedLayout(
                                    inner=Swizzle(3, 4, 3),
                                    offset=0,
                                    outer=Layout((1, 64), (64, 1)),
                                ),
                                shape=(1, 64),
                            )
                            T.copy_async(tile_2, window, tile_3, fill=0)
                        tile_4 = T.tensor_view(
                            T.ptr_of(tile_1[r:r + 1, 0:0 + 64]),
                            layout=Layout((1, 64), (64, 1)),
                            shape=(1, 64),
                        )
                        with Mesh(
                            (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=0,
    outer=Layout((32,), (1,)),
), names=("d0",)
                        ) as threads_1:
                            window_1 = T.tensor_view(
                                T.ptr_of(qs[r:r + 1, 0:0 + 64]),
                                layout=ComposedLayout(
                                    inner=Swizzle(3, 4, 3),
                                    offset=0,
                                    outer=Layout((1, 64), (64, 1)),
                                ),
                                shape=(1, 64),
                            )
                            T.copy_async(tile_4, window_1)
            with scope[1:] as scope_3:
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_2:
                    T.fill(o, 0.0)
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((128,), (1,)),
), names=("d0",)
                ) as threads_3:
                    T.copy(idx, ii_1)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar, 0)
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((128,), (1,)),
), names=("d0",)
                ) as threads_4:
                    T.binary(ii_1, scalar, ii, kind=BinaryKind.ADD)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_1, 0)
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((128,), (1,)),
), names=("d0",)
                ) as threads_5:
                    T.binary(ii, scalar_1, value_2, kind=BinaryKind.GE)
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(scalar_2, 256)
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((128,), (1,)),
), names=("d0",)
                ) as threads_6:
                    T.binary(ii, scalar_2, value_3, kind=BinaryKind.LT)
                    T.binary(value_2, value_3, live, kind=BinaryKind.AND)
                tile_5 = T.tensor_view(
                    T.ptr_of(live[0:0 + 64]), layout=Layout(((8, 8),), ((8, 1),)), shape=(64,)
                )
                tile_6 = T.tensor_view(
                    T.ptr_of(tile_5[0:0 + 64]), layout=Layout((1, 64), (64, 1)), shape=(1, 64)
                )
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_8:
                    T.fill(s_1, 0.0)
                tile_7 = T.tensor_view(
                    T.ptr_of(qs[0:0 + 64, 0:0 + 64]),
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))),
                    ),
                    shape=(64, 64),
                )
                tile_8 = T.tensor_view(
                    T.ptr_of(ks[0:0 + 64, 0:0 + 64]),
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))),
                    ),
                    shape=(64, 64),
                )
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_9:
                    for o_m in range(0, 64, 64):
                        for o_n in range(0, 64, 64):
                            for o_k in range(0, 64, 64):
                                acc_view = T.tensor_view(
                                    T.ptr_of(s_1[o_m:o_m + 64, o_n:o_n + 64]),
                                    layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(tile_7[o_m:o_m + 64, o_k:o_k + 16]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=0,
                                            outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(64, 16),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(tile_8[o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=0,
                                            outer=Layout((16, (8, 8)), (1, (512, 64))),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_9),
                                )
                                acc_view_1 = T.tensor_view(
                                    T.ptr_of(s_1[o_m:o_m + 64, o_n:o_n + 64]),
                                    layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(tile_7[o_m:o_m + 64, o_k + 16:o_k + 16 + 16]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=16,
                                            outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(64, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(tile_8[o_k + 16:o_k + 16 + 16, o_n:o_n + 64]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=16,
                                            outer=Layout((16, (8, 8)), (1, (512, 64))),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_1,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_9),
                                )
                                acc_view_2 = T.tensor_view(
                                    T.ptr_of(s_1[o_m:o_m + 64, o_n:o_n + 64]),
                                    layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(tile_7[o_m:o_m + 64, o_k + 32:o_k + 32 + 16]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=32,
                                            outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(64, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(tile_8[o_k + 32:o_k + 32 + 16, o_n:o_n + 64]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=32,
                                            outer=Layout((16, (8, 8)), (1, (512, 64))),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_2,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_9),
                                )
                                acc_view_3 = T.tensor_view(
                                    T.ptr_of(s_1[o_m:o_m + 64, o_n:o_n + 64]),
                                    layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(tile_7[o_m:o_m + 64, o_k + 48:o_k + 48 + 16]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=48,
                                            outer=Layout(((8, 8), 16), ((512, 64), 1)),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(64, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(tile_8[o_k + 48:o_k + 48 + 16, o_n:o_n + 64]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=48,
                                            outer=Layout((16, (8, 8)), (1, (512, 64))),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_9,
                                    ),
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_3,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_9),
                                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, -1e999)
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_10:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(s_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.where(tile_6, lhs_frame, scalar_3, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(peak[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame,
                        dst_frame_1,
                        axes=(1,),
                        keepdim=True,
                        kind=ReduceKind.MAX,
                    )
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(s[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(peak[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_1, rhs_frame, dst_frame_2, kind=BinaryKind.SUB)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(p_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.unary(src_frame_1, dst_frame_3, kind=UnaryKind.EXP)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(p_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(denom[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    T.reduce(
                        src_frame_2,
                        dst_frame_4,
                        axes=(1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(p_1[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(denom[0:0 + 64, 0:0 + 1]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        shape=(64, 1),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.binary(lhs_frame_2, rhs_frame_1, dst_frame_5, kind=BinaryKind.DIV)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(p[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 8), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.cast(src_frame_3, dst_frame_6, dtype='bf16')
                    tile_9 = T.tensor_view(
                        T.ptr_of(p[0:0 + 64, 0:0 + 64]),
                        layout=((8 @ threads_10.d1, 2, 4 @ threads_10.d0, 4 @ threads_10.d2, 2, 8), (1, 8, 16, 128, 64, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(pr[0:0 + 64, 0:0 + 64]),
                        layout=((4, 8 @ threads_10.d1, 2, 4 @ threads_10.d0, 2, 4 @ threads_10.d2, 2), (1024, 1, 8, 16, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.copy(tile_9, dst_frame_7)
                tile_10 = T.tensor_view(
                    T.ptr_of(ks[0:0 + 64, 0:0 + 64]),
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((8, 8), (8, 8)), ((64, 512), (1, 8))),
                    ),
                    shape=(64, 64),
                )
                tile_11 = T.tensor_view(
                    T.ptr_of(tile_10[0:0 + 64, 0:0 + 64]),
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))),
                    ),
                    shape=(64, 64),
                )
                with Mesh(
                    (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
                ) as threads_19:
                    for o_m_1 in range(0, 64, 64):
                        for o_n_1 in range(0, 64, 64):
                            for o_k_1 in range(0, 64, 16):
                                acc_view_4 = T.tensor_view(
                                    T.ptr_of(o[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 64]),
                                    layout=((8 @ threads_19.d1, 2, 4 @ threads_19.d0, 2, 4 @ threads_19.d2, 8), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 64),
                                )
                                lhs_view_4 = T.tensor_view(
                                    T.ptr_of(pr[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                    layout=((8 @ threads_19.d1, 2, 4 @ threads_19.d0, 2, 4 @ threads_19.d2, 2), (1, 8, 16, 64, 128, 512)),
                                    shape=(64, 16),
                                )
                                rhs_view_4 = T.tensor_view(
                                    T.ptr_of(tile_11[o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 64]),
                                    layout=ShardLayout(
                                        layout=ComposedLayout(
                                            inner=Swizzle(3, 4, 3),
                                            offset=0,
                                            outer=Layout(((2, 8), (1, 64)), ((512, 64), (1024, 1))),
                                        ),
                                        attrs=(B(), B(), B()),
                                        mesh=threads_19,
                                    ),
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view_4,
                                    lhs_view_4,
                                    rhs_view_4,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype='bf16', form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_19),
                                )
                    tile_12 = T.tensor_view(
                        T.ptr_of(o[0:0 + 64, 0:0 + 64]),
                        layout=((2, 4 @ threads_19.d0, 8 @ threads_19.d1, 2, 4 @ threads_19.d2, 8), (8, 16, 1, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(result[0:0 + 64, 0:0 + 64]),
                        layout=((2, 4 @ threads_19.d0, 8 @ threads_19.d1, 2, 4 @ threads_19.d2, 8), (8, 16, 1, 64, 128, 512)),
                        shape=(64, 64),
                    )
                    T.cast(tile_12, dst_frame_8, dtype='bf16')
        with Mesh(
            (Topology("thread", 256),), ComposedLayout(
    inner=None,
    offset=128,
    outer=Layout((4, 8, 4), (32, 4, 1)),
), names=("d0", "d1", "d2")
        ) as threads_22:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 64, 0:0 + 64]),
                layout=((2, 4 @ threads_22.d0, 8 @ threads_22.d1, 2, 4 @ threads_22.d2, 8), (8, 16, 1, 64, 128, 512)),
                shape=(64, 64),
            )
            T.copy(result_view, out)
