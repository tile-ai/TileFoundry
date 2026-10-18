# analysis target=nvidia.h200_sxm module=WGMMA_REPEAT_ALONG_N_ORDER function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r24.00KB/w128.00KB@logical,r24.00KB/w128.00KB@total,r24.00KB/w128.00KB@cta,r24.00KB/w64.25KB@thread;rmem:r448.08KB/w448.00KB@logical,r448.08KB/w448.00KB@total,r448.08KB/w448.00KB@cta,r1.83KB/w1.75KB@thread;smem:r24.00KB/w24.00KB@logical,r24.00KB/w24.00KB@total,r24.00KB/w24.00KB@cta,r4.12KB/w24.00KB@thread footprint=a:4.00KB;b:8.00KB;v12:77:64.00KB footprint-precision=exact peak=gmem:88.00KB;rmem:128.00KB;smem:24.00KB persistent=gmem:24.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 32), "bf16"], b: Tensor[(32, 256), "bf16"], out: Tensor[(128, 256), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 256),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 4, 8, 4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        attrs=(S(0), S(1), S(3), S(6)),
                        mesh=Mesh((Topology("thread", 384),), layout=ComposedLayout(
                            inner=None,
                            offset=128,
                            outer=Layout((2, 4, 8, 4), (128, 32, 4, 1)),
                        ), names=("group", "warp", "lane8", "lane4")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_1:
                T.fill(acc, 0.0)
            lhs_ring_stages = (
                T.tensor_view(16384, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))), shape=(128, 16)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))), shape=(128, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))), shape=(16, 256)),
                T.tensor_view(8192, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))), shape=(16, 256)),
            )
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(T.ptr_of(a), layout=(128, 32), shape=(128, 32))
                    value_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 1, k:k + 1]),
                        layout=((128, 16), (32, 1)),
                        shape=(128, 16),
                    )
                    value_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    T.copy_async_tensor(value_1, value_2)
                    value_3 = T.tensor_view(T.ptr_of(b), layout=(32, 256), shape=(32, 256))
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[k:k + 1, 0:0 + 1]), layout=(16, 256), shape=(16, 256)
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_3:
                    value_6 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        shape=(128, 256),
                    )
                    acc_view = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 128, 0:0 + 64]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_10),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))),
                        shape=(128, 16),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 128, 0:0 + 16]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                        shape=(128, 16),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2]),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_15),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(value_16),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(value_17),
                        layout=Layout(((2, 8), (4, 8, 8)), ((64, 8), (1024, 128, 1))),
                        shape=(16, 256),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 0:0 + 64]),
                        layout=Layout(((2, 8), (8, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 128, 64:64 + 64]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 128, 0:0 + 16]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 64:64 + 64]),
                        layout=Layout(((2, 8), (8, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        lhs_view_1,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_2 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 128, 128:128 + 64]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    lhs_view_2 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 128, 0:0 + 16]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_2 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 128:128 + 64]),
                        layout=Layout(((2, 8), (8, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_2,
                        lhs_view_2,
                        rhs_view_2,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_3 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 128, 192:192 + 64]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 8, 4 @ scope_3.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 64),
                    )
                    lhs_view_3 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 128, 0:0 + 16]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_3 = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 16, 192:192 + 64]),
                        layout=Layout(((2, 8), (8, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 64),
                    )
                    T.tiled_mma(
                        acc_view_3,
                        lhs_view_3,
                        rhs_view_3,
                        atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            value_19 = T.tensor_view(T.ptr_of(out), layout=(128, 256), shape=(128, 256))
            T.fill(value_19, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_20 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_20),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 256),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 256),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 128, 0:0 + 256]),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 8, 4 @ scope_4.lane4, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                    shape=(128, 256),
                )
                T.cast(src_view, result, dtype="bf16")
                value_24 = T.tensor_view(T.ptr_of(out), layout=(128, 256), shape=(128, 256))
                window = T.tensor_view(
                    T.ptr_of(value_24[0:0 + 128, 0:0 + 256]),
                    layout=(128, 256),
                    shape=(128, 256),
                )
                T.copy(result, window)
