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
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
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
                        ), names=("d0", "d1", "d2", "d3")),
                    ),
                    "rmem",
                ]
            )
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
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
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 128, k:k + 16]),
                        layout=((128, 16), (32, 1)),
                        shape=(128, 16),
                    )
                    T.copy_async_tensor(tile, lhs_ring_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 256]), layout=(16, 256), shape=(16, 256)
                    )
                    T.copy_async_tensor(tile_1, rhs_ring_stages[(k // 16) % 2])
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_n in range(0, 256, 64):
                        for o_m in range(0, 128, 128):
                            for o_k in range(0, 16, 16):
                                acc_view = T.tensor_view(
                                    T.ptr_of(acc[o_m:o_m + 128, o_n:o_n + 64]),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 8, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 16) % 2][0:0 + 128, 0:0 + 16]),
                                    layout=Layout(((2, 8, 8), (2, 8)), ((1024, 128, 8), (64, 1))) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(k // 16) % 2][o_k:o_k + 16, o_n:o_n + 64]),
                                    layout=Layout(((2, 8), (8, 8)), ((64, 8), (128, 1))) @ {B()},
                                    shape=(16, 64),
                                )
                                T.tiled_mma(
                                    acc_view,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                result = T.alloc_tensor(
                    tensor_type=Tensor[
                        (128, 256),
                        "bf16",
                        ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 4, 8, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 4096, 512, 128, 64)),
                        "rmem",
                    ]
                )
                T.cast(acc, result, dtype="bf16")
                window = T.tensor_view(
                    T.ptr_of(out[0:0 + 128, 0:0 + 256]), layout=(128, 256), shape=(128, 256)
                )
                T.copy(result, window)
