# analysis target=nvidia.h200_sxm module=SCHED_GDN_PREP_T512 function=gdn_prep topology=cta wave=132/2560
# selection requested=memory executed=memory
# memory traffic=gmem:r30.24KB/w20.34MB@logical,r35.21MB/w50.80GB@total,r30.24KB/w20.34MB@cta,r27.50KB/w20.32MB@thread;rmem:r402.42KB/w369.72KB@logical,r486.52MB/w449.63MB@total,r402.42KB/w369.72KB@cta,r4.12KB/w3.29KB@thread;smem:r32.00KB/w9.50KB@logical,r40.00MB/w11.88MB@total,r32.00KB/w9.50KB@cta,r256B/w9.50KB@thread footprint=a_log:2B;dt_bias:2B;v134:133:8.00KB;v162:147:64B;v177:149:64B;v199:154:1.50KB;v68:114:8.00KB footprint-precision=lower_bound peak=gmem:56.95MB;rmem:58.02KB;smem:4.75KB persistent=gmem:16.33MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gdn_prep(
    proj: Tensor[(512, 16640), "bf16"], conv_w: Tensor[(10240, 4), "bf16", (10240, 4)], a_log: Tensor[(48,), "bf16"], dt_bias: Tensor[(48,), "bf16"], rowidx: Tensor[(608,), "i64"], scratch: Tensor[(2048,), "f32"], scratch_1: Tensor[(2048,), "f32"], scratch_2: Tensor[(16,), "f32"], scratch_3: Tensor[(16,), "f32"], scratch_4: Tensor[(384,), "f32"], out: Tensor[(515, 10336), "f32"]
):
    with Mesh((Topology("cta", 2560),), layout=(32, 80), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 2048]), layout=(16, 128), shape=(16, 128)
        )
        y = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        sf = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        sv = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        xc = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_2 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_3 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_8 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(19, 128), shape=(19, 128)
        )
        wr = T.alloc_tensor(
            tensor_type=Tensor[(4, 128), "f32", ((4, 32, 4), (1, 0, 4)), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(4, 128), "bf16", ((4, 32, 4), (1, 0, 4)), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_17 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_18 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_19 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_20 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_21 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        rr = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_22 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_23 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_24 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        se = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_25 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_26 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        ssum = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        value_27 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "f32", ((4, 4, 1, 1), (4, 1, 0, 0)), "rmem"]
        )
        sq = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_28 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_29 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        with cta[:, :32] as scope:
            with Mesh(
                (Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")
            ) as scope_1:
                tile = T.tensor_view(
                    T.ptr_of(proj[0:0 + 512, (scope.d1 + 0) * 128:(scope.d1 + 0) * 128 + 128]),
                    layout=((512, 128), (16640, 1)),
                    shape=(512, 128),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(rowidx[scope.d0 * 19:scope.d0 * 19 + 19]),
                    layout=(19,),
                    shape=(19,),
                )
                with scope_1[:] as threads:
                    T.copy_async(tile, value_8, tile_1, fill=0.0)
                tile_2 = T.tensor_view(
                    T.ptr_of(value_8[0:0 + 16, 0:0 + 128]), layout=(16, 128), shape=(16, 128)
                )
                with scope_1[:] as threads_1:
                    dst_frame = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_1.d0, 4, 32 @ threads_1.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_2, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_1.d0, 4, 32 @ threads_1.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_1.d0, 4, 32 @ threads_1.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame, dst_frame_1, dtype="f32")
                tile_3 = T.tensor_view(
                    T.ptr_of(conv_w[0:0 + 10240, 0:0 + 4]),
                    layout=((4, 10240), (1, 4)),
                    shape=(4, 10240),
                )
                tile_4 = T.tensor_view(
                    T.ptr_of(tile_3[0:0 + 4, (scope.d1 + 0) * 128:(scope.d1 + 0) * 128 + 128]),
                    layout=((4, 128), (1, 4)),
                    shape=(4, 128),
                )
                with scope_1[:] as threads_3:
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_3.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    T.copy(tile_4, dst_frame_2)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_3.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(wr[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_3.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    T.cast(src_frame_1, dst_frame_3, dtype="f32")
                tile_5 = T.tensor_view(
                    T.ptr_of(wr[0:0 + 1, 0:0 + 128]),
                    layout=((1, 32, 4), (1, 0, 4)),
                    shape=(1, 128),
                )
                with scope_1[:] as threads_5:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_5.d0, 4, 32 @ threads_5.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(acc_3[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_5.d0, 4, 32 @ threads_5.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame, tile_5, dst_frame_4, kind=BinaryKind.MUL)
                tile_6 = T.tensor_view(
                    T.ptr_of(value_8[1:1 + 16, 0:0 + 128]),
                    layout=(16, 128) + 128,
                    shape=(16, 128),
                )
                with scope_1[:] as threads_6:
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_6.d0, 4, 32 @ threads_6.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_6, dst_frame_5)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_6.d0, 4, 32 @ threads_6.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_6.d0, 4, 32 @ threads_6.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_2, dst_frame_6, dtype="f32")
                tile_7 = T.tensor_view(
                    T.ptr_of(wr[1:1 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=1,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_1[:] as threads_8:
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_8.d0, 4, 32 @ threads_8.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_8.d0, 4, 32 @ threads_8.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_1, tile_7, dst_frame_7, kind=BinaryKind.MUL)
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(acc_3[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_8.d0, 4, 32 @ threads_8.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_8.d0, 4, 32 @ threads_8.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(acc_2[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_8.d0, 4, 32 @ threads_8.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_2, rhs_frame, dst_frame_8, kind=BinaryKind.ADD)
                tile_8 = T.tensor_view(
                    T.ptr_of(value_8[2:2 + 16, 0:0 + 128]),
                    layout=(16, 128) + 256,
                    shape=(16, 128),
                )
                with scope_1[:] as threads_10:
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_10.d0, 4, 32 @ threads_10.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_8, dst_frame_9)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_10.d0, 4, 32 @ threads_10.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_10.d0, 4, 32 @ threads_10.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_3, dst_frame_10, dtype="f32")
                tile_9 = T.tensor_view(
                    T.ptr_of(wr[2:2 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=2,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_1[:] as threads_12:
                    lhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_12.d0, 4, 32 @ threads_12.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_12.d0, 4, 32 @ threads_12.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_3, tile_9, dst_frame_11, kind=BinaryKind.MUL)
                    lhs_frame_4 = T.tensor_view(
                        T.ptr_of(acc_2[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_12.d0, 4, 32 @ threads_12.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_12.d0, 4, 32 @ threads_12.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_12 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_12.d0, 4, 32 @ threads_12.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_4, rhs_frame_1, dst_frame_12, kind=BinaryKind.ADD)
                tile_10 = T.tensor_view(
                    T.ptr_of(value_8[3:3 + 16, 0:0 + 128]),
                    layout=(16, 128) + 384,
                    shape=(16, 128),
                )
                with scope_1[:] as threads_14:
                    dst_frame_13 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_14.d0, 4, 32 @ threads_14.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_10, dst_frame_13)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_14.d0, 4, 32 @ threads_14.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_14 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_14.d0, 4, 32 @ threads_14.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_4, dst_frame_14, dtype="f32")
                tile_11 = T.tensor_view(
                    T.ptr_of(wr[3:3 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=3,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_1[:] as threads_16:
                    lhs_frame_5 = T.tensor_view(
                        T.ptr_of(value_17[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_15 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_5, tile_11, dst_frame_15, kind=BinaryKind.MUL)
                    lhs_frame_6 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_16 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_6, rhs_frame_2, dst_frame_16, kind=BinaryKind.ADD)
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_17 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_5, dst_frame_17, dtype="bf16")
                    src_frame_6 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_18 = T.tensor_view(
                        T.ptr_of(xc[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_6, dst_frame_18, dtype="f32")
                    src_frame_7 = T.tensor_view(
                        T.ptr_of(xc[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_19 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.unary(src_frame_7, dst_frame_19, kind=UnaryKind.NEG)
                    src_frame_8 = T.tensor_view(
                        T.ptr_of(value_21[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_20 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_16.d0, 4, 32 @ threads_16.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.unary(src_frame_8, dst_frame_20, kind=UnaryKind.EXP)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                with scope_1[:] as threads_22:
                    lhs_frame_7 = T.tensor_view(
                        T.ptr_of(value_20[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_21 = T.tensor_view(
                        T.ptr_of(value_19[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_7, scalar, dst_frame_21, kind=BinaryKind.ADD)
                    lhs_frame_8 = T.tensor_view(
                        T.ptr_of(xc[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_19[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_22 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_8, rhs_frame_3, dst_frame_22, kind=BinaryKind.DIV)
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_23 = T.tensor_view(
                        T.ptr_of(sv[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_9, dst_frame_23, dtype="bf16")
                    src_frame_10 = T.tensor_view(
                        T.ptr_of(sv[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_24 = T.tensor_view(
                        T.ptr_of(sf[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_10, dst_frame_24, dtype="f32")
                    lhs_frame_9 = T.tensor_view(
                        T.ptr_of(sf[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_4 = T.tensor_view(
                        T.ptr_of(sf[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_25 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_9, rhs_frame_4, dst_frame_25, kind=BinaryKind.MUL)
                    src_frame_11 = T.tensor_view(
                        T.ptr_of(value_29[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_26 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_11, dst_frame_26, dtype="bf16")
                    src_frame_12 = T.tensor_view(
                        T.ptr_of(value_28[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_27 = T.tensor_view(
                        T.ptr_of(sq[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_12, dst_frame_27, dtype="f32")
                    src_frame_13 = T.tensor_view(
                        T.ptr_of(sq[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_22.d0, 4, 32 @ threads_22.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_28 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_22.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.reduce(
                        src_frame_13,
                        dst_frame_28,
                        axes=(-1,),
                        keepdim=True,
                        kind=ReduceKind.SUM,
                    )
                    src_frame_14 = T.tensor_view(
                        T.ptr_of(value_27[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_22.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_29 = T.tensor_view(
                        T.ptr_of(ssum[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_22.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_14, dst_frame_29, dtype="bf16")
                    src_frame_15 = T.tensor_view(
                        T.ptr_of(ssum[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_22.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_30 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_22.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_15, dst_frame_30, dtype="f32")
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 9.98377799987793e-07)
                with scope_1[:] as threads_32:
                    lhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_26[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_31 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(lhs_frame_10, scalar_1, dst_frame_31, kind=BinaryKind.ADD)
                    src_frame_16 = T.tensor_view(
                        T.ptr_of(value_25[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_32 = T.tensor_view(
                        T.ptr_of(se[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_16, dst_frame_32, dtype="bf16")
                    src_frame_17 = T.tensor_view(
                        T.ptr_of(se[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_33 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_17, dst_frame_33, dtype="f32")
                    src_frame_18 = T.tensor_view(
                        T.ptr_of(value_24[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_34 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.unary(src_frame_18, dst_frame_34, kind=UnaryKind.RSQRT)
                    src_frame_19 = T.tensor_view(
                        T.ptr_of(value_23[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_35 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_19, dst_frame_35, dtype="bf16")
                    src_frame_20 = T.tensor_view(
                        T.ptr_of(value_22[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_36 = T.tensor_view(
                        T.ptr_of(rr[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_20, dst_frame_36, dtype="f32")
                    lhs_frame_11 = T.tensor_view(
                        T.ptr_of(sf[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_5 = T.tensor_view(
                        T.ptr_of(rr[0:0 + 16, 0:0 + 1]),
                        layout=((4 @ threads_32.d0, 4, 1, 1), (4, 1, 0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_37 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_11, rhs_frame_5, dst_frame_37, kind=BinaryKind.MUL)
                    src_frame_21 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_38 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_21, dst_frame_38, dtype="bf16")
                    src_frame_22 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_39 = T.tensor_view(
                        T.ptr_of(y[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_22, dst_frame_39, dtype="f32")
                    src_frame_23 = T.tensor_view(
                        T.ptr_of(y[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_32.d0, 4, 32 @ threads_32.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(src_frame_23, value_1)
                window = T.tensor_view(
                    T.ptr_of(out[scope.d0 * 16:scope.d0 * 16 + 16, (scope.d1 + 0) * 128:(scope.d1 + 0) * 128 + 128]),
                    layout=((16, 128), (10336, 1)),
                    shape=(16, 128),
                )
                T.copy(value_1, window)
        value_30_1 = T.tensor_view(
            T.ptr_of(scratch_1[0:0 + 2048]), layout=(16, 128), shape=(16, 128)
        )
        value_31 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        sv_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_32 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        xc_1 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_33 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_4 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_5 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_6 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        acc_7 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_34 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_35 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_36 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(19, 128), shape=(19, 128)
        )
        wr_1 = T.alloc_tensor(
            tensor_type=Tensor[(4, 128), "f32", ((4, 32, 4), (1, 0, 4)), "rmem"]
        )
        value_37 = T.alloc_tensor(
            tensor_type=Tensor[(4, 128), "bf16", ((4, 32, 4), (1, 0, 4)), "rmem"]
        )
        value_38 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_39 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_40 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_41 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_42 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_43 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_44 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_45 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_46 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "bf16", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_47 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_48 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_49 = T.alloc_tensor(
            tensor_type=Tensor[(16, 128), "f32", ((4, 4, 32, 4), (0, 4, 0, 1)), "rmem"]
        )
        value_50_1 = T.tensor_view(T.ptr_of(scratch_2[0:0 + 16]), layout=(16, 1), shape=(16, 1))
        gv = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_51 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
        value_52 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
        value_53 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
        value_54 = T.alloc_tensor(tensor_type=Tensor[(1,), "bf16", (1,), "rmem"])
        sp = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_55 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bool", ((16, 1), (0, 0)), "rmem"]
        )
        av = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_56 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_57 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((16, 1), (0, 0)), "rmem"]
        )
        value_58 = T.alloc_tensor(tensor_type=Tensor[(1,), "f32", (1,), "rmem"])
        value_59 = T.alloc_tensor(tensor_type=Tensor[(1,), "bf16", (1,), "rmem"])
        value_60 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_61 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_62 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_63_1 = T.tensor_view(T.ptr_of(scratch_3[0:0 + 16]), layout=(16, 1), shape=(16, 1))
        be = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_64 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((16, 1), (0, 0)), "rmem"]
        )
        value_65 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_66 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_67 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_68 = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        bv = T.alloc_tensor(tensor_type=Tensor[(16, 1), "f32", ((16, 1), (0, 0)), "rmem"])
        value_69 = T.alloc_tensor(
            tensor_type=Tensor[(16, 1), "bf16", ((16, 1), (0, 0)), "rmem"]
        )
        with cta[:, 32:] as scope_2:
            with Mesh(
                (Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")
            ) as scope_3:
                tile_12 = T.tensor_view(
                    T.ptr_of(proj[0:0 + 512, (scope_2.d1 + 32) * 128:(scope_2.d1 + 32) * 128 + 128]),
                    layout=((512, 128), (16640, 1)),
                    shape=(512, 128),
                )
                tile_13 = T.tensor_view(
                    T.ptr_of(rowidx[scope_2.d0 * 19:scope_2.d0 * 19 + 19]),
                    layout=(19,),
                    shape=(19,),
                )
                with scope_3[:] as threads_42:
                    T.copy_async(tile_12, value_36, tile_13, fill=0.0)
                tile_14 = T.tensor_view(
                    T.ptr_of(value_36[0:0 + 16, 0:0 + 128]), layout=(16, 128), shape=(16, 128)
                )
                with scope_3[:] as threads_43:
                    dst_frame_40 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_43.d0, 4, 32 @ threads_43.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_14, dst_frame_40)
                    src_frame_24 = T.tensor_view(
                        T.ptr_of(value_35[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_43.d0, 4, 32 @ threads_43.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_41 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_43.d0, 4, 32 @ threads_43.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_24, dst_frame_41, dtype="f32")
                tile_15 = T.tensor_view(
                    T.ptr_of(conv_w[0:0 + 10240, 0:0 + 4]),
                    layout=((4, 10240), (1, 4)),
                    shape=(4, 10240),
                )
                tile_16 = T.tensor_view(
                    T.ptr_of(tile_15[0:0 + 4, (scope_2.d1 + 32) * 128:(scope_2.d1 + 32) * 128 + 128]),
                    layout=((4, 128), (1, 4)),
                    shape=(4, 128),
                )
                with scope_3[:] as threads_45:
                    dst_frame_42 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_45.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    T.copy(tile_16, dst_frame_42)
                    src_frame_25 = T.tensor_view(
                        T.ptr_of(value_37[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_45.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    dst_frame_43 = T.tensor_view(
                        T.ptr_of(wr_1[0:0 + 4, 0:0 + 128]),
                        layout=((4, 32 @ threads_45.d1, 4), (1, 0, 4)),
                        shape=(4, 128),
                    )
                    T.cast(src_frame_25, dst_frame_43, dtype="f32")
                tile_17 = T.tensor_view(
                    T.ptr_of(wr_1[0:0 + 1, 0:0 + 128]),
                    layout=((1, 32, 4), (1, 0, 4)),
                    shape=(1, 128),
                )
                with scope_3[:] as threads_47:
                    lhs_frame_12 = T.tensor_view(
                        T.ptr_of(value_34[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_47.d0, 4, 32 @ threads_47.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_44 = T.tensor_view(
                        T.ptr_of(acc_7[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_47.d0, 4, 32 @ threads_47.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_12, tile_17, dst_frame_44, kind=BinaryKind.MUL)
                tile_18 = T.tensor_view(
                    T.ptr_of(value_36[1:1 + 16, 0:0 + 128]),
                    layout=(16, 128) + 128,
                    shape=(16, 128),
                )
                with scope_3[:] as threads_48:
                    dst_frame_45 = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_48.d0, 4, 32 @ threads_48.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_18, dst_frame_45)
                    src_frame_26 = T.tensor_view(
                        T.ptr_of(value_40[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_48.d0, 4, 32 @ threads_48.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_46 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_48.d0, 4, 32 @ threads_48.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_26, dst_frame_46, dtype="f32")
                tile_19 = T.tensor_view(
                    T.ptr_of(wr_1[1:1 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=1,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_3[:] as threads_50:
                    lhs_frame_13 = T.tensor_view(
                        T.ptr_of(value_39[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_50.d0, 4, 32 @ threads_50.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_47 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_50.d0, 4, 32 @ threads_50.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_13, tile_19, dst_frame_47, kind=BinaryKind.MUL)
                    lhs_frame_14 = T.tensor_view(
                        T.ptr_of(acc_7[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_50.d0, 4, 32 @ threads_50.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_50.d0, 4, 32 @ threads_50.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_48 = T.tensor_view(
                        T.ptr_of(acc_6[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_50.d0, 4, 32 @ threads_50.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_14, rhs_frame_6, dst_frame_48, kind=BinaryKind.ADD)
                tile_20 = T.tensor_view(
                    T.ptr_of(value_36[2:2 + 16, 0:0 + 128]),
                    layout=(16, 128) + 256,
                    shape=(16, 128),
                )
                with scope_3[:] as threads_52:
                    dst_frame_49 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_52.d0, 4, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_20, dst_frame_49)
                    src_frame_27 = T.tensor_view(
                        T.ptr_of(value_43[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_52.d0, 4, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_50 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_52.d0, 4, 32 @ threads_52.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_27, dst_frame_50, dtype="f32")
                tile_21 = T.tensor_view(
                    T.ptr_of(wr_1[2:2 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=2,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_3[:] as threads_54:
                    lhs_frame_15 = T.tensor_view(
                        T.ptr_of(value_42[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_54.d0, 4, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_51 = T.tensor_view(
                        T.ptr_of(value_41[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_54.d0, 4, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_15, tile_21, dst_frame_51, kind=BinaryKind.MUL)
                    lhs_frame_16 = T.tensor_view(
                        T.ptr_of(acc_6[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_54.d0, 4, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_7 = T.tensor_view(
                        T.ptr_of(value_41[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_54.d0, 4, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_52 = T.tensor_view(
                        T.ptr_of(acc_5[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_54.d0, 4, 32 @ threads_54.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_16, rhs_frame_7, dst_frame_52, kind=BinaryKind.ADD)
                tile_22 = T.tensor_view(
                    T.ptr_of(value_36[3:3 + 16, 0:0 + 128]),
                    layout=(16, 128) + 384,
                    shape=(16, 128),
                )
                with scope_3[:] as threads_56:
                    dst_frame_53 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_56.d0, 4, 32 @ threads_56.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(tile_22, dst_frame_53)
                    src_frame_28 = T.tensor_view(
                        T.ptr_of(value_46[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_56.d0, 4, 32 @ threads_56.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_54 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_56.d0, 4, 32 @ threads_56.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_28, dst_frame_54, dtype="f32")
                tile_23 = T.tensor_view(
                    T.ptr_of(wr_1[3:3 + 1, 0:0 + 128]),
                    layout=ComposedLayout(
                        inner=None,
                        offset=3,
                        outer=ShardLayout(
                            layout=Layout((1, 32, 4), (1, 0, 4)),
                            attrs=(B(), S(1)),
                            mesh=Mesh((Topology("thread", 128),), layout=ComposedLayout(
                                inner=None,
                                offset=0,
                                outer=Layout((4, 32), (32, 1)),
                            ), names=("warp", "lane")),
                        ),
                    ),
                    shape=(1, 128),
                )
                with scope_3[:] as threads_58:
                    lhs_frame_17 = T.tensor_view(
                        T.ptr_of(value_45[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_55 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_17, tile_23, dst_frame_55, kind=BinaryKind.MUL)
                    lhs_frame_18 = T.tensor_view(
                        T.ptr_of(acc_5[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_8 = T.tensor_view(
                        T.ptr_of(value_44[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_56 = T.tensor_view(
                        T.ptr_of(acc_4[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_18, rhs_frame_8, dst_frame_56, kind=BinaryKind.ADD)
                    src_frame_29 = T.tensor_view(
                        T.ptr_of(acc_4[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_57 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_29, dst_frame_57, dtype="bf16")
                    src_frame_30 = T.tensor_view(
                        T.ptr_of(value_33[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_58 = T.tensor_view(
                        T.ptr_of(xc_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_30, dst_frame_58, dtype="f32")
                    src_frame_31 = T.tensor_view(
                        T.ptr_of(xc_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_59 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.unary(src_frame_31, dst_frame_59, kind=UnaryKind.NEG)
                    src_frame_32 = T.tensor_view(
                        T.ptr_of(value_49[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_60 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_58.d0, 4, 32 @ threads_58.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.unary(src_frame_32, dst_frame_60, kind=UnaryKind.EXP)
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1.0)
                with scope_3[:] as threads_64:
                    lhs_frame_19 = T.tensor_view(
                        T.ptr_of(value_48[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_61 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_19, scalar_2, dst_frame_61, kind=BinaryKind.ADD)
                    lhs_frame_20 = T.tensor_view(
                        T.ptr_of(xc_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    rhs_frame_9 = T.tensor_view(
                        T.ptr_of(value_47[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_62 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.binary(lhs_frame_20, rhs_frame_9, dst_frame_62, kind=BinaryKind.DIV)
                    src_frame_33 = T.tensor_view(
                        T.ptr_of(value_32[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_63 = T.tensor_view(
                        T.ptr_of(sv_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_33, dst_frame_63, dtype="bf16")
                    src_frame_34 = T.tensor_view(
                        T.ptr_of(sv_1[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    dst_frame_64 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.cast(src_frame_34, dst_frame_64, dtype="f32")
                    src_frame_35 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 16, 0:0 + 128]),
                        layout=((4 @ threads_64.d0, 4, 32 @ threads_64.d1, 4), (0, 4, 0, 1)),
                        shape=(16, 128),
                    )
                    T.copy(src_frame_35, value_30_1)
                window_1 = T.tensor_view(
                    T.ptr_of(out[scope_2.d0 * 16:scope_2.d0 * 16 + 16, (scope_2.d1 + 32) * 128:(scope_2.d1 + 32) * 128 + 128]),
                    layout=((16, 128), (10336, 1)),
                    shape=(16, 128),
                )
                T.copy(value_30_1, window_1)
            with Mesh((Topology("thread", 128),), layout=(16,), names=("d0",)) as scope_4:
                tile_24 = T.tensor_view(
                    T.ptr_of(a_log[scope_2.d1:scope_2.d1 + 1]), layout=(1,), shape=(1,)
                )
                with scope_4[:] as threads_69:
                    T.copy(tile_24, value_54)
                    T.cast(value_54, value_53, dtype="f32")
                    T.unary(value_53, value_52, kind=UnaryKind.EXP)
                    T.unary(value_52, value_51, kind=UnaryKind.NEG)
                tile_25 = T.tensor_view(
                    T.ptr_of(proj[scope_2.d0 * 16:scope_2.d0 * 16 + 16, 16432 + scope_2.d1:16432 + scope_2.d1 + 1]),
                    layout=((16, 1), (16640, 1)),
                    shape=(16, 1),
                )
                with scope_4[:] as threads_73:
                    dst_frame_65 = T.tensor_view(
                        T.ptr_of(value_57[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_73.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.copy(tile_25, dst_frame_65)
                    src_frame_36 = T.tensor_view(
                        T.ptr_of(value_57[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_73.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_66 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_73.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_36, dst_frame_66, dtype="f32")
                tile_26 = T.tensor_view(
                    T.ptr_of(dt_bias[scope_2.d1:scope_2.d1 + 1]), layout=(1,), shape=(1,)
                )
                with scope_4[:] as threads_75:
                    T.copy(tile_26, value_59)
                    T.cast(value_59, value_58, dtype="f32")
                    lhs_frame_21 = T.tensor_view(
                        T.ptr_of(value_56[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_75.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_67 = T.tensor_view(
                        T.ptr_of(av[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_75.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(lhs_frame_21, value_58, dst_frame_67, kind=BinaryKind.ADD)
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 20.0)
                with scope_4[:] as threads_78:
                    lhs_frame_22 = T.tensor_view(
                        T.ptr_of(av[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_78.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_68 = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_78.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(lhs_frame_22, scalar_3, dst_frame_68, kind=BinaryKind.GT)
                    src_frame_37 = T.tensor_view(
                        T.ptr_of(av[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_78.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_69 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_78.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.unary(src_frame_37, dst_frame_69, kind=UnaryKind.EXP)
                scalar_4 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_4, 1.0)
                with scope_4[:] as threads_80:
                    lhs_frame_23 = T.tensor_view(
                        T.ptr_of(value_62[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_70 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(lhs_frame_23, scalar_4, dst_frame_70, kind=BinaryKind.ADD)
                    src_frame_38 = T.tensor_view(
                        T.ptr_of(value_61[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_71 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.unary(src_frame_38, dst_frame_71, kind=UnaryKind.LOG)
                    cond_frame = T.tensor_view(
                        T.ptr_of(value_55[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    lhs_frame_24 = T.tensor_view(
                        T.ptr_of(av[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    rhs_frame_10 = T.tensor_view(
                        T.ptr_of(value_60[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_72 = T.tensor_view(
                        T.ptr_of(sp[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.where(cond_frame, lhs_frame_24, rhs_frame_10, dst_frame_72)
                    rhs_frame_11 = T.tensor_view(
                        T.ptr_of(sp[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_73 = T.tensor_view(
                        T.ptr_of(gv[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(value_51, rhs_frame_11, dst_frame_73, kind=BinaryKind.MUL)
                    src_frame_39 = T.tensor_view(
                        T.ptr_of(gv[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_80.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.copy(src_frame_39, value_50_1)
                window_2 = T.tensor_view(
                    T.ptr_of(out[scope_2.d0 * 16:scope_2.d0 * 16 + 16, 10240 + scope_2.d1:10240 + scope_2.d1 + 1]),
                    layout=((16, 1), (10336, 1)),
                    shape=(16, 1),
                )
                T.copy(value_50_1, window_2)
                scalar_5 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_5, 1.0)
                tile_27 = T.tensor_view(
                    T.ptr_of(proj[scope_2.d0 * 16:scope_2.d0 * 16 + 16, 16384 + scope_2.d1:16384 + scope_2.d1 + 1]),
                    layout=((16, 1), (16640, 1)),
                    shape=(16, 1),
                )
                with scope_4[:] as threads_85:
                    dst_frame_74 = T.tensor_view(
                        T.ptr_of(value_69[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.copy(tile_27, dst_frame_74)
                    src_frame_40 = T.tensor_view(
                        T.ptr_of(value_69[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_75 = T.tensor_view(
                        T.ptr_of(bv[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_40, dst_frame_75, dtype="f32")
                    src_frame_41 = T.tensor_view(
                        T.ptr_of(bv[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_76 = T.tensor_view(
                        T.ptr_of(value_68[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.unary(src_frame_41, dst_frame_76, kind=UnaryKind.NEG)
                    src_frame_42 = T.tensor_view(
                        T.ptr_of(value_68[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_77 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_85.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.unary(src_frame_42, dst_frame_77, kind=UnaryKind.EXP)
                scalar_6 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_6, 1.0)
                with scope_4[:] as threads_89:
                    lhs_frame_25 = T.tensor_view(
                        T.ptr_of(value_67[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_78 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(lhs_frame_25, scalar_6, dst_frame_78, kind=BinaryKind.ADD)
                    rhs_frame_12 = T.tensor_view(
                        T.ptr_of(value_66[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_79 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.binary(scalar_5, rhs_frame_12, dst_frame_79, kind=BinaryKind.DIV)
                    src_frame_43 = T.tensor_view(
                        T.ptr_of(value_65[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_80 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_43, dst_frame_80, dtype="bf16")
                    src_frame_44 = T.tensor_view(
                        T.ptr_of(value_64[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    dst_frame_81 = T.tensor_view(
                        T.ptr_of(be[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.cast(src_frame_44, dst_frame_81, dtype="f32")
                    src_frame_45 = T.tensor_view(
                        T.ptr_of(be[0:0 + 16, 0:0 + 1]),
                        layout=((16 @ threads_89.d0, 1), (0, 0)),
                        shape=(16, 1),
                    )
                    T.copy(src_frame_45, value_63_1)
                window_3 = T.tensor_view(
                    T.ptr_of(out[scope_2.d0 * 16:scope_2.d0 * 16 + 16, 10288 + scope_2.d1:10288 + scope_2.d1 + 1]),
                    layout=((16, 1), (10336, 1)),
                    shape=(16, 1),
                )
                T.copy(value_63_1, window_3)
        value_70_1 = T.tensor_view(
            T.ptr_of(scratch_4[0:0 + 384]), layout=(3, 128), shape=(3, 128)
        )
        value_71 = T.alloc_tensor(
            tensor_type=Tensor[(3, 128), "f32", ((3, 128), (1, 0)), "rmem"]
        )
        value_72 = T.alloc_tensor(
            tensor_type=Tensor[(3, 128), "bf16", ((3, 128), (1, 0)), "rmem"]
        )
        with cta[31:] as scope_5:
            with Mesh((Topology("thread", 128),), layout=(128,), names=("d0",)) as scope_6:
                tile_28 = T.tensor_view(
                    T.ptr_of(proj[509:509 + 3, scope_5.d1 * 128:scope_5.d1 * 128 + 128]),
                    layout=((3, 128), (16640, 1)),
                    shape=(3, 128),
                )
                with scope_6[:] as threads_94:
                    dst_frame_82 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 3, 0:0 + 128]),
                        layout=((3, 128 @ threads_94.d0), (1, 0)),
                        shape=(3, 128),
                    )
                    T.copy(tile_28, dst_frame_82)
                    src_frame_46 = T.tensor_view(
                        T.ptr_of(value_72[0:0 + 3, 0:0 + 128]),
                        layout=((3, 128 @ threads_94.d0), (1, 0)),
                        shape=(3, 128),
                    )
                    dst_frame_83 = T.tensor_view(
                        T.ptr_of(value_71[0:0 + 3, 0:0 + 128]),
                        layout=((3, 128 @ threads_94.d0), (1, 0)),
                        shape=(3, 128),
                    )
                    T.cast(src_frame_46, dst_frame_83, dtype="f32")
                    src_frame_47 = T.tensor_view(
                        T.ptr_of(value_71[0:0 + 3, 0:0 + 128]),
                        layout=((3, 128 @ threads_94.d0), (1, 0)),
                        shape=(3, 128),
                    )
                    T.copy(src_frame_47, value_70_1)
                window_4 = T.tensor_view(
                    T.ptr_of(out[512:512 + 3, scope_5.d1 * 128:scope_5.d1 * 128 + 128]),
                    layout=((3, 128), (10336, 1)),
                    shape=(3, 128),
                )
                T.copy(value_70_1, window_4)
