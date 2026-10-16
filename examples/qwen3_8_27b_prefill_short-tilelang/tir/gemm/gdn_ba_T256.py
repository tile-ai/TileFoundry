# analysis target=nvidia.h200_sxm module=SCHED_GDN_BA_T256 function=gemm topology=cta wave=96/96
# selection requested=memory executed=memory
# memory traffic=gmem:r188.00KB/w784.00KB@logical,r17.62MB/w73.50MB@total,r188.00KB/w784.00KB@cta,r188.00KB/w776.03KB@thread;rmem:r88.37KB/w88.02KB@logical,r8.28MB/w8.25MB@total,r88.37KB/w88.02KB@cta,r728B/w368B@thread;smem:r180.00KB/w180.00KB@logical,r16.88MB/w16.88MB@total,r180.00KB/w180.00KB@cta,r7.50KB/w180.00KB@thread footprint=a:256.00KB;v22:53:8.00KB;v24:55:768.00KB;w:96.00KB footprint-precision=exact peak=gmem:4.20MB;rmem:8.02KB;smem:36.00KB persistent=gmem:3.44MB
#   buffer=a holds=1.10MB time=none space=cta.bn reuse=1.25MB fits=yes precision=exact
#   buffer=w holds=1.10MB time=none space=cta.bm reuse=96.00KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 5120), "bf16"], w: Tensor[(96, 5120), "bf16"], scratch: Tensor[(2048,), "f32"], out: Tensor[(8, 256, 96), "f32"]
):
    with Mesh((Topology("cta", 96),), layout=(8, 2, 6), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
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
                T.tensor_view(4096, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(16, 64)),
                T.tensor_view(2048, dtype="bf16", storage="smem", layout=Layout(((2, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(16, 64)),
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
                        T.ptr_of(w[cta.d2 * 16:cta.d2 * 16 + 1, (cta.d0 * 640) + kk:(cta.d0 * 640) + kk + 1]),
                        layout=((16, 64), (5120, 1)),
                        shape=(16, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(kk // 64) % 2]),
                        layout=Layout(((2, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(16, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 16, 16):
                            for o_k in range(0, 64, 64):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 2, 4 @ scope_3.d3, 2), (1024, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 16),
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
                                    layout=((64, 16), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 16),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 16]),
                                    layout=(Layout((16, (2, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 16),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 16]),
                                    layout=(Layout((16, (2, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 16),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 16]),
                                    layout=(Layout((16, (2, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 16),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 16]),
                                    layout=(Layout((16, (2, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 16),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                value_7 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((1, 2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 2, 4 @ scope_4.d3, 2), (0, 1024, 16, 8, 1, 512, 128, 64)),
                    shape=(1, 128, 16),
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 2048]), layout=(1, 128, 16), shape=(1, 128, 16)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(1, 128, 16), shape=(1, 128, 16)
                )
                T.copy(value_7, value_8)
                value_9 = T.tensor_view(T.ptr_of(out), layout=(8, 256, 96), shape=(8, 256, 96))
                window = T.tensor_view(
                    T.ptr_of(value_9[cta.d0:cta.d0 + 1, cta.d1 * 128:cta.d1 * 128 + 128, cta.d2 * 16:cta.d2 * 16 + 16]),
                    layout=((1, 128, 16), (24576, 96, 1)),
                    shape=(1, 128, 16),
                )
                T.copy(value_8, window)
