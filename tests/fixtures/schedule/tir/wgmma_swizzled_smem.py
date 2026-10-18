# analysis target=nvidia.h200_sxm module=WGMMA_SWIZZLED_SMEM function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w8.00KB@logical,r6.00KB/w8.00KB@total,r6.00KB/w8.00KB@cta,r6.00KB/w4.03KB@thread;rmem:r28.08KB/w28.00KB@logical,r28.08KB/w28.00KB@total,r28.08KB/w28.00KB@cta,r304B/w224B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r640B/w6.00KB@thread footprint=a:2.00KB;b:1.00KB;v12:77:4.00KB footprint-precision=exact peak=gmem:10.00KB;rmem:8.00KB;smem:6.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 32), "bf16"], out: Tensor[(64, 32), "bf16"]
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
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (1, 32)), ((256, 32), (512, 1))) | Swizzle(2, 4, 3), shape=(16, 32)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((2, 8), (1, 32)), ((256, 32), (512, 1))) | Swizzle(2, 4, 3), shape=(16, 32)),
            )
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    T.copy_async_tensor(value, lhs_ring_stages[(k // 16) % 2])
                    value_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(16, 32), shape=(16, 32)
                    )
                    T.copy_async_tensor(value_1, rhs_ring_stages[(k // 16) % 2])
                with Mesh(
                    scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")
                ) as scope_3:
                    acc_view = T.tensor_view(
                        T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                        layout=((4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 4, 4 @ scope_3.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        shape=(64, 32),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 2][0:0 + 64, 0:0 + 16]),
                        layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                        shape=(64, 16),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 2][0:0 + 16, 0:0 + 32]),
                        layout=(Layout(((2, 8), (1, 32)), ((256, 32), (512, 1))) | Swizzle(2, 4, 3)) @ {B()},
                        shape=(16, 32),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            T.fill(out, 0.0)
            with Mesh(scope[1:], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 32),
                        "bf16",
                        ((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                src_view = T.tensor_view(
                    T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                    layout=((4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 4, 4 @ scope_4.lane4, 2), (16, 8, 1, 512, 128, 64)),
                    shape=(64, 32),
                )
                T.cast(src_view, result, dtype="bf16")
                T.copy(result, out)
