# analysis target=nvidia.h200_sxm module=SCHED_DOWN_T128 function=gemm topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r3.49MB/w5.08MB@logical,r447.00MB/w650.00MB@total,r3.49MB/w5.08MB@cta,r3.49MB/w5.04MB@thread;rmem:r5.36MB/w5.35MB@logical,r685.54MB/w685.00MB@total,r5.36MB/w5.35MB@cta,r25.71KB/w21.42KB@thread;smem:r3.45MB/w3.45MB@logical,r442.00MB/w442.00MB@total,r3.45MB/w3.45MB@cta,r374.00KB/w3.45MB@thread footprint=a:32.00KB;v22:53:40.00KB;v24:55:5.00MB;w:1.25MB footprint-precision=exact peak=gmem:179.29MB;rmem:40.02KB;smem:208.00KB persistent=gmem:174.25MB
#   buffer=a holds=6.32MB time=none space=cta.bn reuse=1.97MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(128, 17408), "bf16"], w: Tensor[(5120, 17408), "bf16"], scratch: Tensor[(10240,), "f32"], out: Tensor[(2, 128, 5120), "f32"]
):
    with Mesh((Topology("cta", 128),), layout=(2, 64, 1), names=("d0", "d1", "d2")) as cta:
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (128, 80),
                    "f32",
                    ShardLayout(
                        layout=Layout((2, 4, 2, 8, 10, 4, 2), (5120, 16, 8, 1, 512, 128, 64)),
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
                T.tensor_view(147456, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(163840, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(180224, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
                T.tensor_view(196608, dtype="bf16", storage="smem", layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(128, 64)),
            )
            rhs_ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(10240, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(20480, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(30720, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(40960, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(51200, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(61440, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
                T.tensor_view(71680, dtype="bf16", storage="smem", layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), shape=(80, 64)),
            )
            for kk in range(0, 8704, 64):
                with scope[:1, :32] as scope_2:
                    value = T.tensor_view(
                        T.ptr_of(a[0:0 + 1, (cta.d0 * 8704) + kk:(cta.d0 * 8704) + kk + 1]),
                        layout=((128, 64), (17408, 1)),
                        shape=(128, 64),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(lhs_ring_stages[(kk // 64) % 8]),
                        layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(128, 64),
                    )
                    T.copy_async_tensor(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(w[cta.d1 * 80:cta.d1 * 80 + 1, (cta.d0 * 8704) + kk:(cta.d0 * 8704) + kk + 1]),
                        layout=((80, 64), (17408, 1)),
                        shape=(80, 64),
                    )
                    value_3 = T.tensor_view(
                        T.ptr_of(rhs_ring_stages[(kk // 64) % 8]),
                        layout=Layout(((10, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3),
                        shape=(80, 64),
                    )
                    T.copy_async_tensor(value_2, value_3)
                with Mesh(
                    scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
                ) as scope_3:
                    for o_m in range(0, 128, 128):
                        for o_n in range(0, 80, 80):
                            for o_k in range(0, 64, 64):
                                value_4 = T.tensor_view(
                                    T.ptr_of(acc),
                                    layout=((2 @ scope_3.d0, 4 @ scope_3.d1, 2, 8 @ scope_3.d2, 10, 4 @ scope_3.d3, 2), (5120, 16, 8, 1, 512, 128, 64)),
                                    shape=(128, 80),
                                )
                                value_5 = T.tensor_view(
                                    T.ptr_of(lhs_ring_stages[(kk // 64) % 8]),
                                    layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3),
                                    shape=(128, 64),
                                )
                                lhs_view = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k:o_k + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                value_6 = T.tensor_view(
                                    T.ptr_of(rhs_ring_stages[(kk // 64) % 8]),
                                    layout=((64, 80), (1, 64)) | Swizzle(3, 4, 3),
                                    shape=(64, 80),
                                )
                                rhs_view = T.tensor_view(
                                    T.ptr_of(value_6[o_k:o_k + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view,
                                    rhs_view,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 16:o_k + 16 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_1 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 16:o_k + 16 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 16 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_1,
                                    rhs_view_1,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 32:o_k + 32 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_2 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 32:o_k + 32 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 32 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_2,
                                    rhs_view_2,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
                                lhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_5[o_m:o_m + 128, o_k + 48:o_k + 48 + 16]),
                                    layout=(Layout(((2, 8, 8), 16), ((4096, 512, 64), 1)) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(128, 16),
                                )
                                rhs_view_3 = T.tensor_view(
                                    T.ptr_of(value_6[o_k + 48:o_k + 48 + 16, o_n:o_n + 80]),
                                    layout=(Layout((16, (10, 8)), (1, (512, 64))) + 48 | Swizzle(3, 4, 3)) @ {B()},
                                    shape=(16, 80),
                                )
                                T.tiled_mma(
                                    value_4,
                                    lhs_view_3,
                                    rhs_view_3,
                                    atom=T.cuda.sm90.Wgmma(n=80, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K),
                                )
            T.fill(out, 0.0)
            with Mesh(
                scope[1:], layout=(2, 4, 8, 4), names=("d0", "d1", "d2", "d3")
            ) as scope_4:
                value_7 = T.tensor_view(
                    T.ptr_of(acc),
                    layout=((1, 2 @ scope_4.d0, 4 @ scope_4.d1, 2, 8 @ scope_4.d2, 10, 4 @ scope_4.d3, 2), (0, 5120, 16, 8, 1, 512, 128, 64)),
                    shape=(1, 128, 80),
                )
                tile_dst_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 10240]), layout=(1, 128, 80), shape=(1, 128, 80)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(tile_dst_1), layout=(1, 128, 80), shape=(1, 128, 80)
                )
                T.copy(value_7, value_8)
                value_9 = T.tensor_view(
                    T.ptr_of(out), layout=(2, 128, 5120), shape=(2, 128, 5120)
                )
                window = T.tensor_view(
                    T.ptr_of(value_9[cta.d0:cta.d0 + 1, cta.d2 * 128:cta.d2 * 128 + 128, cta.d1 * 80:cta.d1 * 80 + 80]),
                    layout=((1, 128, 80), (655360, 5120, 1)),
                    shape=(1, 128, 80),
                )
                T.copy(value_8, window)
