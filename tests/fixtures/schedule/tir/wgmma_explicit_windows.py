# analysis target=nvidia.h200_sxm module=WGMMA_EXPLICIT_WINDOWS function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r12.00KB/w32.00KB@logical,r24.00KB/w32.00KB@total,r24.00KB/w32.00KB@cta,r24.00KB/w16.12KB@thread;rmem:r112.19KB/w88.00KB@logical,r112.31KB/w112.00KB@total,r112.31KB/w112.00KB@cta,r1.19KB/w896B@thread;smem:r24.00KB/w12.00KB@logical,r24.00KB/w24.00KB@total,r24.00KB/w24.00KB@cta,r2.50KB/w24.00KB@thread footprint=a:2.00KB;b:1.00KB;v12:85:4.00KB footprint-precision=exact peak=gmem:28.00KB;rmem:8.00KB;smem:6.00KB persistent=gmem:12.00KB
#   buffer=a holds=10.00KB time=n space=none reuse=4.00KB fits=yes precision=exact
#   buffer=b holds=16.00KB time=m space=none reuse=4.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 32), "bf16"], b: Tensor[(32, 64), "bf16", ((32, 64), (1, 32))], out: Tensor[(128, 64), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 1), (128, 8))), shape=(16, 32)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 1), (128, 8))), shape=(16, 32)),
            )
            for m in range(0, 128, 64):
                for n in range(0, 64, 32):
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
                                ), names=("d0", "d1", "d2")),
                            ),
                            "rmem",
                        ]
                    )
                    with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_1:
                        T.fill(acc, 0.0)
                    for k in range(0, 32, 16):
                        with scope[:1, :32] as scope_2:
                            tile = T.tensor_view(
                                T.ptr_of(a[m:m + 64, k:k + 16]),
                                layout=((64, 16), (32, 1)),
                                shape=(64, 16),
                            )
                            T.copy_async_tensor(
                                tile,
                                lhs_ring_stages[(((((m // 64) * 2) + (n // 32)) * 2) + (k // 16)) % 2],
                            )
                            tile_1 = T.tensor_view(
                                T.ptr_of(b[k:k + 16, n:n + 32]),
                                layout=((16, 32), (1, 32)),
                                shape=(16, 32),
                            )
                            T.copy_async_tensor(
                                tile_1,
                                rhs_ring_stages[(((((m // 64) * 2) + (n // 32)) * 2) + (k // 16)) % 2],
                            )
                        with Mesh(
                            scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")
                        ) as scope_3:
                            for o_m in range(0, 64, 64):
                                for o_n in range(0, 32, 32):
                                    for o_k in range(0, 16, 16):
                                        lhs_view = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 64) * 2) + (n // 32)) * 2) + (k // 16)) % 2][0:0 + 64, 0:0 + 16]),
                                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))) @ {B()},
                                            shape=(64, 16),
                                        )
                                        rhs_view = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 64) * 2) + (n // 32)) * 2) + (k // 16)) % 2][0:0 + 16, 0:0 + 32]),
                                            layout=Layout(((2, 8), (4, 8)), ((64, 1), (128, 8))) @ {B()},
                                            shape=(16, 32),
                                        )
                                        T.tiled_mma(
                                            acc,
                                            lhs_view,
                                            rhs_view,
                                            atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                        )
                    with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_4:
                        tile_out = T.alloc_tensor(
                            tensor_type=Tensor[
                                (64, 32),
                                "bf16",
                                ((4 @ scope_4.d0, 2, 8 @ scope_4.d1, 4, 4 @ scope_4.d2, 2), (16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        T.cast(acc, tile_out, dtype="bf16")
                        window = T.tensor_view(
                            T.ptr_of(out[m:m + 64, n:n + 32]),
                            layout=((64, 32), (64, 1)),
                            shape=(64, 32),
                        )
                        T.copy(tile_out, window)
