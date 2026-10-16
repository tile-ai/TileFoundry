# analysis target=nvidia.h200_sxm module=SCHED_DOWN_T256 function=gemm topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r4.86MB/w10.16MB@logical,r622.00MB/w1.27GB@total,r4.86MB/w10.16MB@cta,r4.86MB/w10.08MB@thread;rmem:r10.71MB/w10.70MB@logical,r1.34GB/w1.34GB@total,r10.71MB/w10.70MB@cta,r47.12KB/w42.83KB@thread;smem:r4.78MB/w4.78MB@logical,r612.00MB/w612.00MB@total,r4.78MB/w4.78MB@cta,r714.00KB/w4.78MB@thread footprint=a:64.00KB;v22:53:80.00KB;v24:55:10.00MB;w:1.25MB footprint-precision=exact peak=gmem:188.58MB;rmem:80.02KB;smem:144.00KB persistent=gmem:178.50MB
#   buffer=a holds=11.39MB time=none space=cta.bn reuse=1.94MB fits=yes precision=exact
#   buffer=w holds=11.39MB time=none space=cta.bm reuse=1.25MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(256, 17408), "bf16"], w: Tensor[(5120, 17408), "bf16"], scratch: Tensor[(20480,), "f32"], out: Tensor[(2, 256, 5120), "f32"]
):
    with Mesh((Topology("cta", 128),), layout=(2, 32, 2), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 160),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 20, 4, 2), (10240, 16, 8, 1, 512, 128, 64)),
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
                T.tensor_view(81920, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(98304, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(114688, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(131072, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(40960, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
                T.tensor_view(61440, dtype="bf16", storage="smem", layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(160, 64)),
            )
            for kk in range(0, 8704, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[cta.d2 * 128:cta.d2 * 128 + 1, (cta.d0 * 8704) + kk:(cta.d0 * 8704) + kk + 1]),
                        layout=((128, 64), (17408, 1)),
                        shape=(128, 64),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(kk // 64) % 4]),
                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w[cta.d1 * 160:cta.d1 * 160 + 1, (cta.d0 * 8704) + kk:(cta.d0 * 8704) + kk + 1]),
                        layout=((160, 64), (17408, 1)),
                        shape=(160, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(kk // 64) % 4]),
                        layout=Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(160, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 160, 160):
                            for o_k in range(0, 64, 64):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 20, 4 @ scope_3.d3, 2), (10240, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 160),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(kk // 64) % 4]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(kk // 64) % 4]),
                                    layout=((64, 160), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 160),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 160]),
                                    layout=(Layout((16, (20, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 160),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 160]),
                                    layout=(Layout((16, (20, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 160),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 160]),
                                    layout=(Layout((16, (20, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 160),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 160]),
                                    layout=(Layout((16, (20, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 160),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                value_7 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((1, 2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 20, 4 @ scope_4.d3, 2), (0, 10240, 16, 8, 1, 512, 128, 64)),
                    shape=(1, 128, 160),
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 20480]), layout=(1, 128, 160), shape=(1, 128, 160)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(1, 128, 160), shape=(1, 128, 160)
                )
                T.copy(value_7, value_8)
                value_9 = T.tensor_view(
                    T.ptr_of(out), layout=(2, 256, 5120), shape=(2, 256, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_9[cta.d0:cta.d0 + 1, cta.d2 * 128:cta.d2 * 128 + 128, cta.d1 * 160:cta.d1 * 160 + 160]),
                    layout=((1, 128, 160), (1310720, 5120, 1)),
                    shape=(1, 128, 160),
                )
                T.copy(value_8, window)
