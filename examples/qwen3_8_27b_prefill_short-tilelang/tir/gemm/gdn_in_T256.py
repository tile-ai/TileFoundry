# analysis target=nvidia.h200_sxm module=SCHED_GDN_IN_T256 function=gemm topology=cta wave=130/130
# selection requested=memory executed=memory
# memory traffic=gmem:r3.81MB/w8.25MB@logical,r495.62MB/w1.05GB@total,r3.81MB/w8.25MB@cta,r3.81MB/w8.25MB@thread;rmem:r10.19MB/w10.19MB@logical,r1.29GB/w1.29GB@total,r10.19MB/w10.19MB@cta,r43.30KB/w40.77KB@thread;smem:r3.81MB/w3.81MB@logical,r495.62MB/w495.62MB@total,r3.81MB/w3.81MB@cta,r724.00KB/w3.75MB@thread footprint=a:32.00KB;v19:54:64.00KB;v21:56:8.12MB;w:2.03MB footprint-precision=exact peak=gmem:173.19MB;rmem:128.02KB;smem:192.00KB persistent=gmem:165.00MB
#   buffer=w holds=10.25MB time=none space=cta.bm reuse=2.03MB fits=yes precision=exact
#   buffer=a holds=10.25MB time=none space=cta.bn reuse=2.00MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 5120), "bf16"], w: Tensor[(16640, 5120), "bf16"], scratch: Tensor[(32768,), "bf16"], out: Tensor[(256, 16640), "bf16"]
):
    with Mesh((Topology("cta", 130),), layout=(65, 2), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
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
            lhs_ring_stages = (
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(32768, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(65536, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(256, 64)),
            )
            for k in range(0, 5120, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 1, k:k + 1]),
                        layout=((128, 64), (5120, 1)),
                        shape=(128, 64),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(k // 64) % 4]),
                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w[cta.d0 * 256:cta.d0 * 256 + 1, k:k + 1]),
                        layout=((256, 64), (5120, 1)),
                        shape=(256, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(k // 64) % 4]),
                        layout=Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(256, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 256, 256):
                            for o_k in range(0, 64, 64):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 32, 4 @ scope_3.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 256),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(k // 64) % 4]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(k // 64) % 4]),
                                    layout=((64, 256), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 256),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 256]),
                                    layout=(Layout((16, (32, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 256),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
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
                value_7 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 32, 4 @ scope_4.d3, 2), (16384, 16, 8, 1, 512, 128, 64)),
                    shape=(128, 256),
                )
                T.cast(value_7, tile_out, dtype="bf16")
                staged_dst = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 256), shape=(128, 256)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(staged_dst), layout=(128, 256), shape=(128, 256)
                )
                T.copy(tile_out, value_8)
            with scope[:1, :32] as scope_5:
                staged_dst_2 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(128, 256), shape=(128, 256)
                )
                value_9 = T.tensor_view(
                    T.ptr_of(staged_dst_2), layout=(128, 256), shape=(128, 256)
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 32768]), layout=(128, 256), shape=(128, 256)
                )
                value_10 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(128, 256), shape=(128, 256)
                )
                T.copy_async_tensor(value_9, value_10)
                value_11 = T.tensor_view(T.ptr_of(out), layout=(256, 16640), shape=(256, 16640))
                window = T.tensor_view(
                    T.ptr_of(value_11[cta.d1 * 128:cta.d1 * 128 + 128, cta.d0 * 256:cta.d0 * 256 + 256]),
                    layout=((128, 256), (16640, 1)),
                    shape=(128, 256),
                )
                T.copy(value_10, window)
