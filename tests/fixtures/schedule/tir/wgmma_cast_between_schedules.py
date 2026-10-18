# analysis target=nvidia.h200_sxm module=WGMMA_CAST_BETWEEN_SCHEDULES function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16.00KB/w256B@logical,r16.00KB/w256B@total,r16.00KB/w256B@cta,r4.19KB/w132B@thread;rmem:r62.95KB/w54.88KB@logical,r62.95KB/w54.88KB@total,r62.95KB/w54.88KB@cta,r708B/w596B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r640B/w4.06KB@thread footprint=a:2.00KB;b_f32:2.00KB;bias:8.00KB;v21:131:128B footprint-precision=exact peak=gmem:16.12KB;rmem:16.00KB;smem:6.00KB persistent=gmem:16.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b_f32: Tensor[(32, 32), "f32"], bias: Tensor[(64, 32), "f32"], out: Tensor[(64, 1), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 256),), layout=(2, 128), names=("role", "participant")
        ) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 4, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_1:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
            )
            for k in range(0, 32, 16):
                with Mesh((Topology("thread", 256),), layout=(32,), names=("lane",)) as scope_2:
                    value = T.tensor_view(T.ptr_of(a), layout=(64, 32), shape=(64, 32))
                    value_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 1, k:k + 1]),
                        layout=((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    T.copy_async_tensor(value_1, value_2)
                with Mesh((Topology("thread", 256),), layout=(32,), names=("lane",)) as scope_3:
                    b_tile_dst = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 32), "f32", ((16, 32 @ scope_3.lane), (1, 16)), "rmem"
                        ]
                    )
                    value_3 = T.tensor_view(T.ptr_of(b_f32), layout=(32, 32), shape=(32, 32))
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[k:k + 1, 0:0 + 1]), layout=(16, 32), shape=(16, 32)
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(b_tile_dst),
                        layout=((16, 32 @ scope_3.lane), (1, 16)),
                        shape=(16, 32),
                    )
                    T.copy(value_4, value_5)
                    b = T.alloc_tensor(
                        tensor_type=Tensor[
                            (16, 32), "bf16", ((16, 32 @ scope_3.lane), (1, 16)), "rmem"
                        ]
                    )
                    T.cast(value_5, b, dtype="bf16")
                    value_6 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    T.copy(b, value_6)
                with Mesh(
                    scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_4:
                    value_7 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(value_8),
                        layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    acc_view = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 64, 0:0 + 32]),
                        layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_10),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 64, 0:0 + 16]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                        shape=(64, 16),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(value_13),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 16, 0:0 + 32]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 32),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            value_16 = T.tensor_view(T.ptr_of(out), layout=(64, 1), shape=(64, 1))
            T.fill(value_16, 0.0)
            acc_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 4, 4, 2), (16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(2), S(4)),
                        mesh=Mesh((Topology("thread", 256),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((4, 8, 4), (32, 4, 1)),
                        ), names=("warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_5:
                bias_r_dst = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "f32",
                        ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_17 = T.tensor_view(T.ptr_of(bias), layout=(64, 32), shape=(64, 32))
                value_18 = T.tensor_view(
                    T.ptr_of(bias_r_dst),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.copy(value_17, value_18)
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "f32",
                        ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_20 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_20),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                lhs_view_1 = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 64, 0:0 + 32]),
                    layout=((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 4, 4 @ scope_5.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.binary(lhs_view_1, value_18, value_19, kind=BinaryKind.ADD)
                T.relu(value_19, acc_1)
                explicit_dst = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((8 @ scope_5.lane8, 2, 4 @ scope_5.warp, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                value_24 = T.tensor_view(
                    T.ptr_of(explicit_dst),
                    layout=((8 @ scope_5.lane8, 2, 4 @ scope_5.warp, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                    shape=(64, 1),
                )
                T.reduce(acc_1, value_24, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                automatic = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_5.warp, 2, 8 @ scope_5.lane8, 1, 1, 1), (16, 8, 1, 0, 0, 0)),
                        "rmem",
                    ]
                )
                T.reduce(acc_1, automatic, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                value_25 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "f32",
                        ((4 @ scope_5.warp, 8 @ scope_5.lane8, 2, 1), (16, 2, 1, 1)),
                        "rmem",
                    ]
                )
                T.binary(value_24, automatic, value_25, kind=BinaryKind.ADD)
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 1),
                        "bf16",
                        ((4 @ scope_5.warp, 8 @ scope_5.lane8, 2, 1), (16, 2, 1, 1)),
                        "rmem",
                    ]
                )
                T.cast(value_25, result, dtype="bf16")
                value_26 = T.tensor_view(T.ptr_of(out), layout=(64, 1), shape=(64, 1))
                window = T.tensor_view(
                    T.ptr_of(value_26[0:0 + 64, 0:0 + 1]), layout=(64, 1), shape=(64, 1)
                )
                T.copy(result, window)
