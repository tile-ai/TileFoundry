# analysis target=nvidia.h200_sxm module=GEMM_8192X17408X5120_TMA_STORE function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r522.00MB/w816.00MB@logical,r16.20GB/w816.00MB@total,r16.20GB/w816.00MB@cta,r16.20GB/w816.00MB@thread;rmem:r43.30GB/w42.77GB@logical,r43.31GB/w43.30GB@total,r43.31GB/w43.30GB@cta,r183.88MB/w173.19MB@thread;smem:r16.20GB/w522.00MB@logical,r16.20GB/w16.20GB@total,r16.20GB/w16.20GB@cta,r3.00GB/w15.94GB@thread footprint=a:16.00KB;b:32.00KB;v14:92:64.00KB;v16:94:64.00KB footprint-precision=exact peak=gmem:522.00MB;rmem:128.00KB;smem:208.00KB persistent=gmem:250.00MB
#   buffer=b holds=175.56MB time=m space=none reuse=10.46GB fits=no precision=exact
#   buffer=a holds=3.88MB time=n space=none reuse=83.75MB fits=yes precision=exact
#   error="l2 reuse window m holds 175.56MB at a 1-unit wave, exceeding capacity 47.68MB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(8192, 5120), "bf16"], b: Tensor[(5120, 17408), "bf16"], out: Tensor[(8192, 17408), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            lhs_ring_stages = (
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((4, 2, 8), (4, 64)), ((4096, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), shape=(64, 256)),
            )
            for m in range(0, 8192, 128):
                for n in range(0, 17408, 256):
                    acc = T.alloc_tensor(
                        tensor_type=Tensor[
                            (128, 256),
                            "f32",
                            ShardLayout(
                                layout=Layout((2, 4, 2, 8, 32, 4, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                attrs=(S(0), S(1), S(3), S(5)),
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
                    for k in range(0, 5120, 64):
                        with scope[:1, :32] as scope_2:
                            tile = T.tensor_view(
                                T.ptr_of(a[m:m + 128, k:k + 64]),
                                layout=((128, 64), (5120, 1)),
                                shape=(128, 64),
                            )
                            T.copy_async_tensor(
                                tile,
                                lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3],
                            )
                            tile_1 = T.tensor_view(
                                T.ptr_of(b[k:k + 64, n:n + 256]),
                                layout=((64, 256), (17408, 1)),
                                shape=(64, 256),
                            )
                            T.copy_async_tensor(
                                tile_1,
                                rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3],
                            )
                        with Mesh(
                            scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                        ) as scope_3:
                            for o_m in range(0, 128, 128):
                                for o_n in range(0, 256, 256):
                                    for o_k in range(0, 64, 64):
                                        lhs_view = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k:o_k + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_k:o_k + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            acc,
                                            lhs_view,
                                            rhs_view,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_1 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_1 = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            acc,
                                            lhs_view_1,
                                            rhs_view_1,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_2 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_2 = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            acc,
                                            lhs_view_2,
                                            rhs_view_2,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                                        lhs_view_3 = T.tensor_view(
                                            T.ptr_of(lhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                            layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(128, 16),
                                        )
                                        rhs_view_3 = T.tensor_view(
                                            T.ptr_of(rhs_ring_stages[(((((m // 128) * 68) + (n // 256)) * 80) + (k // 64)) % 3][o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                            layout=(Layout(((2, 8), (4, 64)), ((512, 64), (1024, 1))) | Swizzle(3, 4, 3)) @ {B()},
                                            shape=(16, 256),
                                        )
                                        T.tiled_mma(
                                            acc,
                                            lhs_view_3,
                                            rhs_view_3,
                                            atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN),
                                        )
                    with Mesh(
                        scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                    ) as scope_4:
                        tile_out = T.alloc_tensor(
                            tensor_type=Tensor[
                                (128, 256),
                                "bf16",
                                ((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                "rmem",
                            ]
                        )
                        T.cast(acc, tile_out, dtype="bf16")
                        staged_dst = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        T.copy(tile_out, staged_dst)
                    with scope[:1, :32] as scope_5:
                        staged_dst_2 = T.tensor_view(
                            147456,
                            dtype="bf16",
                            storage="smem",
                            layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3),
                            shape=(128, 256),
                        )
                        window = T.tensor_view(
                            T.ptr_of(out[m:m + 128, n:n + 256]),
                            layout=((128, 256), (17408, 1)),
                            shape=(128, 256),
                        )
                        T.copy_async_tensor(staged_dst_2, window)
