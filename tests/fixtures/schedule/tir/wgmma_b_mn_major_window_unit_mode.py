# analysis target=nvidia.h200_sxm module=WindowedMnMajorB function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r10.00KB/w64.00KB@logical,r10.00KB/w64.00KB@total,r10.00KB/w64.00KB@cta,r6.00KB/w32.12KB@thread;rmem:r64.03KB/w64.00KB@logical,r64.03KB/w64.00KB@total,r64.03KB/w64.00KB@cta,r288B/w256B@thread;smem:r6.00KB/w10.00KB@logical,r6.00KB/w10.00KB@total,r6.00KB/w10.00KB@cta,r576B/w6.00KB@thread footprint=a:2.00KB;b:8.00KB;v8:22:32.00KB footprint-precision=exact peak=gmem:42.00KB;rmem:32.00KB;smem:10.00KB persistent=gmem:10.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 256), "bf16"], out: Tensor[(64, 128), "f32"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh(
            (Topology("thread", 256),), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
        ) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(64, 128), shape=(64, 128))
            T.fill(value, 0.0)
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (64, 128),
                    "f32",
                    ((4 @ scope.warp, 2, 8 @ scope.lane8, 2 @ scope.group, 8, 4 @ scope.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                    "rmem",
                ]
            )
            T.fill(acc, 0.0)
            value_2 = T.tensor_view(T.ptr_of(a), layout=(64, 16), shape=(64, 16))
            value_1 = T.tensor_view(
                8192,
                dtype="bf16",
                storage="smem",
                layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                shape=(64, 16),
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1),
                layout=Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3),
                shape=(64, 16),
            )
            T.copy(value_2, value_3)
            value_5 = T.tensor_view(T.ptr_of(b), layout=(16, 256), shape=(16, 256))
            value_4 = T.tensor_view(
                0,
                dtype="bf16",
                storage="smem",
                layout=Layout(((2, 8), (2, 1, 2, 64)), ((512, 64), (8192, 1024, 4096, 1))) | Swizzle(3, 4, 3),
                shape=(16, 256),
            )
            value_6 = T.tensor_view(
                T.ptr_of(value_4),
                layout=ShardLayout(
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((2, 8), (2, 1, 2, 64)), ((512, 64), (8192, 1024, 4096, 1))),
                    ),
                    attrs=(S(4), B(), B(), B()),
                    mesh=scope,
                ),
                shape=(16, 256),
            )
            T.copy(value_5, value_6)
            acc_view = T.tensor_view(
                T.ptr_of(acc[0:0 + 64, 0:0 + 128]),
                layout=((4 @ scope.warp, 2, 8 @ scope.lane8, 2 @ scope.group, 8, 4 @ scope.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                shape=(64, 128),
            )
            lhs_view = T.tensor_view(
                T.ptr_of(value_3[0:0 + 64, 0:0 + 16]),
                layout=(Layout(((8, 8), 16), ((512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                shape=(64, 16),
            )
            value_7 = T.tensor_view(
                T.ptr_of(value_6),
                layout=Layout((16, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3),
                shape=(16, 128),
            )
            rhs_view = T.tensor_view(
                T.ptr_of(value_7[0:0 + 16, 0:0 + 128]),
                layout=ShardLayout(
                    layout=ComposedLayout(
                        inner=Swizzle(3, 4, 3),
                        offset=0,
                        outer=Layout(((2, 8), (1, 2, 64)), ((512, 64), (1024, 4096, 1))),
                    ),
                    attrs=(S(3), B(), B(), B()),
                    mesh=scope,
                ),
                shape=(16, 128),
            )
            T.tiled_mma(
                acc_view,
                lhs_view,
                rhs_view,
                atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
            )
            value_8 = T.tensor_view(
                T.ptr_of(acc),
                layout=((4 @ scope.warp, 2, 8 @ scope.lane8, 2 @ scope.group, 8, 4 @ scope.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                shape=(64, 128),
            )
            value_view = T.tensor_view(
                T.ptr_of(value_8[0:0 + 64, 0:0 + 128]),
                layout=((4 @ scope.warp, 2, 8 @ scope.lane8, 2 @ scope.group, 8, 4 @ scope.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)),
                shape=(64, 128),
            )
            value_9 = T.tensor_view(T.ptr_of(out), layout=(64, 128), shape=(64, 128))
            window = T.tensor_view(
                T.ptr_of(value_9[0:0 + 64, 0:0 + 128]), layout=(64, 128), shape=(64, 128)
            )
            T.copy(value_view, window)
