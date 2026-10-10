# analysis target=nvidia.h200_sxm module=WGMMA_RS_A_FROM_ACCUMULATOR function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w0@logical,r6.00KB/w0@total,r6.00KB/w0@cta,r6.00KB/w0@thread;rmem:r30.08KB/w34.00KB@logical,r30.08KB/w34.00KB@total,r30.08KB/w34.00KB@cta,r320B/w272B@thread;smem:r6.00KB/w6.00KB@logical,r6.00KB/w6.00KB@total,r6.00KB/w6.00KB@cta,r448B/w6.00KB@thread footprint=a:2.00KB;b:512B;b1:1.00KB footprint-precision=exact peak=gmem:6.00KB;rmem:10.00KB;smem:5.00KB persistent=gmem:6.00KB

from __future__ import annotations

from tilefoundry.dsl import *  # noqa: F401, F403
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    a: Tensor[(64, 32), "bf16"], b: Tensor[(32, 16), "bf16"], b1: Tensor[(16, 32), "bf16"], out: Tensor[(64, 32), "bf16"]
):
    with Mesh((Topology("cta", 1),), Layout((1,), (1,)), names=("d0",)) as cta:
        p = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 16), "f32", Layout((8, 2, 4, 2, 4, 2), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        value = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 16), "bf16", Layout((8, 2, 4, 2, 4, 2), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        acc = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "f32", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[
                (64, 32), "bf16", Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)), "rmem"
            ]
        )
        with Mesh(
            (Topology("thread", 256),), Layout((2, 128), (128, 1)), names=("d0", "d1")
        ) as scope:
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as threads:
                T.fill(p, 0.0)
            lhs_stages = (T.tensor_view(1024, dtype='bf16', storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)), T.tensor_view(3072, dtype='bf16', storage="smem", layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), shape=(64, 16)))
            rhs_stages = (T.tensor_view(0, dtype='bf16', storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)), T.tensor_view(512, dtype='bf16', storage="smem", layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))), shape=(16, 16)))
            for k in range(0, 32, 16):
                with scope[:1, :32] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 64, k:k + 16]),
                        layout=Layout((64, 16), (32, 1)),
                        shape=(64, 16),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 16]),
                        layout=Layout((16, 16), (16, 1)),
                        shape=(16, 16),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 16) % 2])
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_2:
                    with scope_2 as threads_3:
                        for o_m in range(0, 64, 64):
                            for o_n in range(0, 16, 16):
                                for o_k in range(0, 16, 16):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(p[o_m:o_m + 64, o_n:o_n + 16]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2, 4 @ threads_3.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_stages[(k // 16) % 2][o_m:o_m + 64, o_k:o_k + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(64, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_stages[(k // 16) % 2][o_k:o_k + 16, o_n:o_n + 16]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (2, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_3,
                                        ),
                                        shape=(16, 16),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm90.Wgmma(n=16, dtype='bf16', form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_3),
                                    )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_3:
                with scope_3 as threads_4:
                    src_frame = T.tensor_view(
                        T.ptr_of(p[0:0 + 64, 0:0 + 16]),
                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 16),
                    )
                    dst_frame = T.tensor_view(
                        T.ptr_of(value[0:0 + 64, 0:0 + 16]),
                        layout=((8 @ threads_4.d1, 2, 4 @ threads_4.d0, 2, 4 @ threads_4.d2, 2), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 16),
                    )
                    T.cast(src_frame, dst_frame, dtype='bf16')
                    T.fill(acc, 0.0)
            rhs_1_stages = (T.tensor_view(0, dtype='bf16', storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)), T.tensor_view(1024, dtype='bf16', storage="smem", layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))), shape=(16, 32)))
            for j in range(0, 16, 16):
                with scope[:1, :32] as scope_4:
                    tile_2 = T.tensor_view(
                        T.ptr_of(b1[j:j + 16, 0:0 + 32]),
                        layout=Layout((16, 32), (32, 1)),
                        shape=(16, 32),
                    )
                    with Mesh(scope_4, layout=(32,), names=("d0",)) as threads_6:
                        T.copy_async_tensor(tile_2, rhs_1_stages[(j // 16) % 2])
                with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_5:
                    with scope_5 as threads_7:
                        for o_m_1 in range(0, 64, 64):
                            for o_n_1 in range(0, 32, 32):
                                for o_k_1 in range(0, 16, 16):
                                    acc_view_1 = T.tensor_view(
                                        T.ptr_of(acc[o_m_1:o_m_1 + 64, o_n_1:o_n_1 + 32]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 4), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 32),
                                    )
                                    lhs_view_1 = T.tensor_view(
                                        T.ptr_of(value[o_m_1:o_m_1 + 64, o_k_1:o_k_1 + 16]),
                                        layout=((8 @ threads_7.d1, 2, 4 @ threads_7.d0, 2, 4 @ threads_7.d2, 2), (1, 8, 16, 64, 128, 512)),
                                        shape=(64, 16),
                                    )
                                    rhs_view_1 = T.tensor_view(
                                        T.ptr_of(rhs_1_stages[(j // 16) % 2][o_k_1:o_k_1 + 16, o_n_1:o_n_1 + 32]),
                                        layout=ShardLayout(
                                            layout=Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                            attrs=(B(), B(), B()),
                                            mesh=threads_7,
                                        ),
                                        shape=(16, 32),
                                    )
                                    T.tiled_mma(
                                        acc_view_1,
                                        lhs_view_1,
                                        rhs_view_1,
                                        atom=T.cuda.sm90.Wgmma(n=32, dtype='bf16', form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN, mesh=threads_7),
                                    )
            with Mesh(scope[1:], layout=(4, 8, 4), names=("d0", "d1", "d2")) as scope_6:
                with scope_6 as threads_8:
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                        layout=((8 @ threads_8.d1, 2, 4 @ threads_8.d0, 2, 4 @ threads_8.d2, 4), (1, 8, 16, 64, 128, 512)),
                        shape=(64, 32),
                    )
                    T.cast(src_frame_1, dst_frame_1, dtype='bf16')
        with Mesh(
            (Topology("thread", 256),), Layout((4, 8, 4), (32, 4, 1)) + 128, names=("d0", "d1", "d2")
        ) as threads_9:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 64, 0:0 + 32]),
                layout=((8 @ threads_9.d1, 2, 4 @ threads_9.d0, 2, 4 @ threads_9.d2, 4), (1, 8, 16, 64, 128, 512)),
                shape=(64, 32),
            )
            T.copy(result_view, out)
