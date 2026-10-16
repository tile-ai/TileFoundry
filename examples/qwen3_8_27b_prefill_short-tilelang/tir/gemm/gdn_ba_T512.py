# analysis target=nvidia.h200_sxm module=SCHED_GDN_BA_T512 function=gemm topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r216.00KB/w1.53MB@logical,r20.25MB/w147.00MB@total,r216.00KB/w1.53MB@cta,r216.00KB/w1.52MB@thread;rmem:r176.37KB/w176.02KB@logical,r16.53MB/w16.50MB@total,r176.37KB/w176.02KB@cta,r1.05KB/w720B@thread;smem:r200.00KB/w200.00KB@logical,r18.75MB/w18.75MB@total,r200.00KB/w200.00KB@cta,r12.50KB/w200.00KB@thread footprint=a:512.00KB;v22:53:16.00KB;v24:55:1.50MB;w:96.00KB footprint-precision=exact peak=gmem:7.45MB;rmem:16.02KB;smem:40.00KB persistent=gmem:5.94MB
#   buffer=a holds=2.11MB time=none space=cta.bn reuse=1.00MB fits=yes precision=exact
#   buffer=w holds=2.11MB time=none space=cta.bm reuse=288.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(512, 5120), "bf16"], w: Tensor[(96, 5120), "bf16"], scratch: Tensor[(4096,), "f32"], out: Tensor[(8, 512, 96), "f32"]
):
    with Mesh((Topology("cta", 96),), layout=(8, 4, 3), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 32),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 4, 4, 2), (2048, 16, 8, 1, 512, 128, 64)),
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
                T.tensor_view(8192, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(24576, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((4, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(32, 64)),
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((4, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(32, 64)),
            )
            for kk in range(0, 640, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[cta.d1 * 128:cta.d1 * 128 + 1, (cta.d0 * 640) + kk:(cta.d0 * 640) + kk + 1]),
                        layout=((128, 64), (5120, 1)),
                        shape=(128, 64),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(kk // 64) % 2]),
                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w[cta.d2 * 32:cta.d2 * 32 + 1, (cta.d0 * 640) + kk:(cta.d0 * 640) + kk + 1]),
                        layout=((32, 64), (5120, 1)),
                        shape=(32, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(kk // 64) % 2]),
                        layout=Layout(((4, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(32, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 32, 32):
                            for o_k in range(0, 64, 64):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 4, 4 @ scope_3.d3, 2), (2048, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 32),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(kk // 64) % 2]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(kk // 64) % 2]),
                                    layout=((64, 32), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 32),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 32]),
                                    layout=(Layout((16, (4, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 32]),
                                    layout=(Layout((16, (4, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 32]),
                                    layout=(Layout((16, (4, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 32]),
                                    layout=(Layout((16, (4, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 32),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                value_7 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((1, 2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 4, 4 @ scope_4.d3, 2), (0, 2048, 16, 8, 1, 512, 128, 64)),
                    shape=(1, 128, 32),
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 4096]), layout=(1, 128, 32), shape=(1, 128, 32)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(1, 128, 32), shape=(1, 128, 32)
                )
                T.copy(value_7, value_8)
                value_9 = T.tensor_view(T.ptr_of(out), layout=(8, 512, 96), shape=(8, 512, 96))
                window = T.tensor_view(
                    T.ptr_of(value_9[cta.d0:cta.d0 + 1, cta.d1 * 128:cta.d1 * 128 + 128, cta.d2 * 32:cta.d2 * 32 + 32]),
                    layout=((1, 128, 32), (49152, 96, 1)),
                    shape=(1, 128, 32),
                )
                T.copy(value_8, window)
