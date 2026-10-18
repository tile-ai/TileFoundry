# analysis target=nvidia.h200_sxm module=WGMMA_TMA_3STAGE function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r18.00KB/w8.00KB@logical,r18.00KB/w8.00KB@total,r18.00KB/w8.00KB@cta,r18.00KB/w4.02KB@thread;rmem:r44.14KB/w44.00KB@logical,r44.14KB/w44.00KB@total,r44.14KB/w44.00KB@cta,r320B/w176B@thread;smem:r18.00KB/w18.00KB@logical,r18.00KB/w18.00KB@total,r18.00KB/w18.00KB@cta,r768B/w18.00KB@thread footprint=a:4.00KB;b:512B;v12:75:4.00KB footprint-precision=exact peak=gmem:22.00KB;rmem:8.00KB;smem:13.50KB persistent=gmem:18.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 64), "bf16"], b: Tensor[(64, 16), "bf16"], out: Tensor[(128, 16), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 384),), layout=(3, 128), names=("warpgroup", "participant")
        ) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 16),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 2, 4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        attrs=(S(0), S(1), S(3), S(5)),
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
                T.tensor_view(1536, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))), shape=(128, 16)),
                T.tensor_view(5632, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))), shape=(128, 16)),
                T.tensor_view(9728, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))), shape=(128, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)),
                T.tensor_view(512, dtype="bf16", storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)),
            )
            for k in range(0, 64, 16):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((128, 16), (64, 1)),
                        shape=(128, 16),
                    )
                    T.copy_async_tensor(value, lhs_ring_stages[(k // 16) % 3])
                    value_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(16, 16), shape=(16, 16)
                    )
                    T.copy_async_tensor(value_1, rhs_ring_stages[(k // 16) % 3])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_3:
                    acc_view = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 16) % 3][0:0 + 128, 0:0 + 16]),
                        layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 16) % 3][0:0 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 16),
                        "bf16",
                        ((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        "rmem",
                    ]
                )
                src_view = T.tensor_view(
                    T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                    layout=((2 @ scope_4.group, 4 @ scope_4.warp, 2, 8 @ scope_4.lane8, 2, 4 @ scope_4.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 16),
                )
                T.cast(src_view, result, dtype="bf16")
                T.copy(result, out)
