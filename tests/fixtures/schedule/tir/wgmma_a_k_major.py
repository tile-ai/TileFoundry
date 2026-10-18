# analysis target=nvidia.h200_sxm module=WGMMA_A_K_MAJOR function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r10.00KB/w12.00KB@logical,r10.00KB/w12.00KB@total,r10.00KB/w12.00KB@cta,r10.00KB/w8.03KB@thread;rmem:r28.08KB/w28.00KB@logical,r28.08KB/w28.00KB@total,r28.08KB/w28.00KB@cta,r304B/w224B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r640B/w6.00KB@thread footprint=at:2.00KB;b:1.00KB;v13:78:4.00KB;v2:43:4.00KB footprint-precision=exact peak=gmem:10.00KB;rmem:8.00KB;smem:6.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    at: Tensor[(64, 64), "fp8e4m3"], b: Tensor[(64, 32), "fp8e4m3", ((64, 32), (1, 64))], scratch: Tensor[(2048,), "fp8e4m3"], out: Tensor[(64, 32), "bf16"]
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
                T.tensor_view(2048, dtype="fp8e4m3", storage="smem", layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)),
                T.tensor_view(4096, dtype="fp8e4m3", storage="smem", layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)),
                T.tensor_view(1024, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)),
            )
            for k in range(0, 64, 32):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(T.ptr_of(at), layout=(64, 64), shape=(64, 64))
                    value_1 = T.tensor_view(
                        T.ptr_of(value[k:k + 1, 0:0 + 1]), layout=(32, 64), shape=(32, 64)
                    )
                    tile = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 32, 0:0 + 64]),
                        layout=((64, 32), (1, 64)),
                        shape=(64, 32),
                    )
                    a_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 2048]), layout=(64, 32), shape=(64, 32)
                    )
                    T.copy(tile, a_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 32) % 2]),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    T.copy_async_tensor(a_1, value_2)
                    value_3 = T.tensor_view(
                        T.ptr_of(b), layout=((64, 32), (1, 64)), shape=(64, 32)
                    )
                    value_4 = T.tensor_view(
                        T.ptr_of(value_3[k:k + 1, 0:0 + 1]),
                        layout=((32, 32), (1, 64)),
                        shape=(32, 32),
                    )
                    value_5 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 32) % 2]),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    T.copy_async_tensor(value_4, value_5)
                with Mesh(
                    scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_3:
                    value_6 = T.tensor_view(
                        T.ptr_of(acc),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_7 = T.tensor_view(
                        T.ptr_of(value_6),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_8 = T.tensor_view(
                        T.ptr_of(value_7),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    acc_view = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 64, 0:0 + 32]),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    value_9 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 32) % 2]),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    value_10 = T.tensor_view(
                        T.ptr_of(value_9),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    value_11 = T.tensor_view(
                        T.ptr_of(value_10),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    value_12 = T.tensor_view(
                        T.ptr_of(value_11),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    value_13 = T.tensor_view(
                        T.ptr_of(value_12),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                        shape=(64, 32),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 64, 0:0 + 32]),
                        layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))) @ {B()},
                        shape=(64, 32),
                    )
                    value_14 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 32) % 2]),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    value_15 = T.tensor_view(
                        T.ptr_of(value_14),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    value_16 = T.tensor_view(
                        T.ptr_of(value_15),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    value_17 = T.tensor_view(
                        T.ptr_of(value_16),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    value_18 = T.tensor_view(
                        T.ptr_of(value_17),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                        shape=(32, 32),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(value_18[0:0 + 32, 0:0 + 32]),
                        layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))) @ {B()},
                        shape=(32, 32),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                    )
            value_19 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
            T.fill(value_19, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "bf16",
                        ((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                value_20 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_21 = T.tensor_view(
                    T.ptr_of(value_20),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_22 = T.tensor_view(
                    T.ptr_of(value_21),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                value_23 = T.tensor_view(
                    T.ptr_of(value_22),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 64, 0:0 + 32]),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.cast(src_view, result, dtype="bf16")
                value_24 = T.tensor_view(T.ptr_of(out), layout=(64, 32), shape=(64, 32))
                window = T.tensor_view(
                    T.ptr_of(value_24[0:0 + 64, 0:0 + 32]), layout=(64, 32), shape=(64, 32)
                )
                T.copy(result, window)
