# analysis target=nvidia.h200_sxm module=WGMMA_A_K_MAJOR function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r10.00KB/w4.00KB@logical,r10.00KB/w4.00KB@total,r10.00KB/w4.00KB@cta,r10.00KB/w4.00KB@thread;rmem:r24.06KB/w28.00KB@logical,r24.06KB/w28.00KB@total,r24.06KB/w28.00KB@cta,r256B/w224B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r640B/w6.00KB@thread footprint=at:2.00KB;b:1.00KB;v2:43:4.00KB footprint-precision=exact peak=gmem:8.00KB;rmem:8.00KB;smem:6.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    at: Tensor[(64, 64), "fp8e4m3"], b: Tensor[(64, 32), "fp8e4m3", ((64, 32), (1, 64))], scratch: Tensor[(2048,), "fp8e4m3"], out: Tensor[(64, 32), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", ((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "bf16", ((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        with Mesh((Topology("thread", 256),), layout=(2, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                T.fill(acc, 0.0)
            lhs_stages = (
                T.tensor_view(2048, dtype="fp8e4m3", storage="smem", layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)),
                T.tensor_view(4096, dtype="fp8e4m3", storage="smem", layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))), shape=(64, 32)),
            )
            rhs_stages = (
                T.tensor_view(0, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)),
                T.tensor_view(1024, dtype="fp8e4m3", storage="smem", layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))), shape=(32, 32)),
            )
            for k in range(0, 64, 32):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(at[k:k + 32, 0:0 + 64]), layout=(32, 64), shape=(32, 64)
                    )
                    a_1 = T.tensor_view(
                        T.ptr_of(scratch[0:0 + 2048]), layout=(64, 32), shape=(64, 32)
                    )
                    tile_1 = T.tensor_view(
                        T.ptr_of(tile[0:0 + 32, 0:0 + 64]),
                        layout=((64, 32), (1, 64)),
                        shape=(64, 32),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy(tile_1, a_1)
                        T.copy_async_tensor(a_1, lhs_stages[(k // 32) % 2])
                    tile_2 = T.tensor_view(
                        T.ptr_of(b[k:k + 32, 0:0 + 32]),
                        layout=((32, 32), (1, 64)),
                        shape=(32, 32),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_3:
                        T.copy_async_tensor(tile_2, rhs_stages[(k // 32) % 2])
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_2:
                    with scope_2 as threads_4:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 32, 32):
                                for o_k in range(0, 32, 32):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 64, o_n:o_n + 32]),
                                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 4), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 32),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 32) % 2][o_m:o_m + 64, o_k:o_k + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((8, 8), (2, 16)), ((256, 16), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(64, 32),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 32) % 2][o_k:o_k + 32, o_n:o_n + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 16), (4, 8)), ((128, 1), (256, 16))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_4,
                                        ),
                                        shape=(32, 32),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=32, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K, mesh=threads_4),
                                    )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                with scope_3 as threads_5:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_5.d1, 2, 4 @ threads_5.d0, 2, 4 @ threads_5.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.cast(src_frame, dst_frame, dtype="bf16")
        with Mesh(
            (Topology("thread", 256),), layout=(4, 8, 4) + 128, names=("d0", "d1", "d2")
        ) as threads_6:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                layout=((8 @ threads_6.d1, 2, 4 @ threads_6.d0, 2, 4 @ threads_6.d2, 4), (1, 8, 16, 64, 128, 512)),
                shape=(64, 32),
            )
            T.copy(result_view, out)
