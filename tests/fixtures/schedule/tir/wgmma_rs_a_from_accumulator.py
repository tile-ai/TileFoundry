# analysis target=nvidia.h200_sxm module=WGMMA_RS_A_FROM_ACCUMULATOR function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w8.00KB@logical,r6.00KB/w8.00KB@total,r6.00KB/w8.00KB@cta,r6.00KB/w4.03KB@thread;rmem:r34.09KB/w34.00KB@logical,r34.09KB/w34.00KB@total,r34.09KB/w34.00KB@cta,r368B/w272B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r448B/w6.00KB@thread footprint=a:2.00KB;b:512B;b1:1.00KB;v20:108:4.00KB footprint-precision=exact peak=gmem:10.00KB;rmem:10.00KB;smem:5.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 16), "bf16"], b1: Tensor[(16, 32), "bf16"], out: Tensor[(64, 32), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 256),), layout=(2, 128), names=("role", "participant")
        ) as scope:
            p = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 2, 4, 2), (16, 8, 1, 512, 128, 64)),
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
                T.fill(p, 0.0)
            lhs_ring_stages = (
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
                T.tensor_view(3072, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)),
                T.tensor_view(512, dtype="bf16", storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)),
            )
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_2:
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
                    value_3 = T.tensor_view(T.ptr_of(b), layout=(32, 16), shape=(32, 16))
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[k:k + 1, 0:0 + 1]), layout=(16, 16), shape=(16, 16)
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_3:
                    value_6 = T.tensor_view(
                        T.ptr_of(p),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    acc_view = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 64, 0:0 + 16]),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_9),
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
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                        shape=(64, 16),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 64, 0:0 + 16]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                        shape=(64, 16),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_15),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(value_16),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(value_17),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                        shape=(16, 16),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            value_19 = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 16),
                    "bf16",
                    ShardLayout(
                        layout=Layout((4, 2, 8, 2, 4, 2), (16, 8, 1, 512, 128, 64)),
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
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_4:
                value_20 = T.tensor_view(
                    T.ptr_of(p),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_20),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 64, 0:0 + 16]),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 16),
                )
                T.cast(src_view, value_19, dtype="bf16")
                T.fill(acc, 0.0)
            rhs_ring_1_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)),
            )
            for j in range(0, 16, 16):
                with scope[:1, :32] as scope_5:
                    value_24 = T.tensor_view(T.ptr_of(b1), layout=(16, 32), shape=(16, 32))
                    value_25 = T.tensor_view(
                        T.ptr_of(value_24), layout=(16, 32), shape=(16, 32)
                    )
                    value_26 = T.tensor_view(
                        T.ptr_of(rhs_ring_1_stages[(j // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    T.copy_async_tensor(value_25, value_26)
                with Mesh(
                    scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_6:
                    value_27 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_28 = T.tensor_view(
                        T.ptr_of(value_27),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_29 = T.tensor_view(
                        T.ptr_of(value_28),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_30 = T.tensor_view(
                        T.ptr_of(value_29),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_31 = T.tensor_view(
                        T.ptr_of(value_30),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(value_31[0:0 + 64, 0:0 + 32]),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 4, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_32 = T.tensor_view(
                        T.ptr_of(value_19),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 2, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_33 = T.tensor_view(
                        T.ptr_of(value_32),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 2, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_34 = T.tensor_view(
                        T.ptr_of(value_33),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 2, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_35 = T.tensor_view(
                        T.ptr_of(value_34),
                        layout=((4 @ scope_6.warp, 2, 8 @ scope_6.lane8, 2, 4 @ scope_6.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 16),
                    )
                    value_36 = T.tensor_view(
                        T.ptr_of(rhs_ring_1_stages[(j // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    value_37 = T.tensor_view(
                        T.ptr_of(value_36),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    value_38 = T.tensor_view(
                        T.ptr_of(value_37),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                        shape=(16, 32),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_38[0:0 + 16, 0:0 + 32]),
                        layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 32),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        value_35,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            value_39 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
            T.fill(value_39, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_7:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "bf16",
                        ((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_40 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_41 = T.tensor_view(
                    T.ptr_of(value_40),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_42 = T.tensor_view(
                    T.ptr_of(value_41),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_43 = T.tensor_view(
                    T.ptr_of(value_42),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_44 = T.tensor_view(
                    T.ptr_of(value_43),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_45 = T.tensor_view(
                    T.ptr_of(value_44),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_45[0:0 + 64, 0:0 + 32]),
                    layout=((4 @ scope_7.warp, 2, 8 @ scope_7.lane8, 4, 4 @ scope_7.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.cast(src_view_1, result, dtype="bf16")
                value_46 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
                window = T.tensor_view(
                    T.ptr_of(value_46[0:0 + 64, 0:0 + 32]), layout=(64, 32), shape=(64, 32)
                )
                T.copy(result, window)
