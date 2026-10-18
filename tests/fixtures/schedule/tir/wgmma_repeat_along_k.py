# analysis target=nvidia.h200_sxm module=WGMMA_REPEAT_ALONG_K function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r36.00KB/w8.00KB@logical,r36.00KB/w8.00KB@total,r36.00KB/w8.00KB@cta,r36.00KB/w4.02KB@thread;rmem:r28.08KB/w28.00KB@logical,r28.08KB/w28.00KB@total,r28.08KB/w28.00KB@cta,r192B/w112B@thread;smem:r36.00KB/w36.00KB@logical,r36.00KB/w36.00KB@total,r36.00KB/w36.00KB@cta,r1.50KB/w36.00KB@thread footprint=a:16.00KB;b:2.00KB;v12:78:4.00KB footprint-precision=exact peak=gmem:40.00KB;rmem:8.00KB;smem:36.00KB persistent=gmem:36.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 128), "bf16"], b: Tensor[(128, 16), "bf16"], out: Tensor[(128, 16), "bf16"]
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
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 8)), ((256, 64, 8), (128, 1))), shape=(64, 16)),
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (2, 8)), ((256, 64, 8), (128, 1))), shape=(64, 16)),
            )
            for k in range(0, 128, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, k:k + 1]),
                        layout=((128, 64), (128, 1)),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, lhs_ring_stages[(k // 64) % 2])
                    value_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 1, 0:0 + 1]), layout=(64, 16), shape=(64, 16)
                    )
                    T.copy_async_tensor(value_1, rhs_ring_stages[(k // 64) % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
                ) as scope_3:
                    acc_view = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    lhs_view = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 2][0:0 + 128, 0:0 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 2][0:0 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view,
                        lhs_view,
                        rhs_view,
                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_1 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    lhs_view_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 2][0:0 + 128, 16:16 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_1 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 2][16:16 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view_1,
                        lhs_view_1,
                        rhs_view_1,
                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_2 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    lhs_view_2 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 2][0:0 + 128, 32:32 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_2 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 2][32:32 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view_2,
                        lhs_view_2,
                        rhs_view_2,
                        atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                    )
                    acc_view_3 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 128, 0:0 + 16]),
                        layout=((2 @ scope_3.group, 4 @ scope_3.warp, 2, 8 @ scope_3.lane8, 2, 4 @ scope_3.lane4, 2), (1024, 16, 8, 1, 512, 128, 64)),
                        shape=(128, 16),
                    )
                    lhs_view_3 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 2][0:0 + 128, 48:48 + 16]),
                        layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                        shape=(128, 16),
                    )
                    rhs_view_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 2][48:48 + 16, 0:0 + 16]),
                        layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))) @ {B()},
                        shape=(16, 16),
                    )
                    T.tiled_mma(
                        acc_view_3,
                        lhs_view_3,
                        rhs_view_3,
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
