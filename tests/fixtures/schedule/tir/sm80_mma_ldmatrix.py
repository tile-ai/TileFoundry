# analysis target=nvidia.h200_sxm module=SM80_MMA_LDMATRIX function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r1.50KB/w0@logical,r1.50KB/w0@total,r1.50KB/w0@cta,r1.50KB/w0@thread;rmem:r3.06KB/w3.25KB@logical,r3.06KB/w3.25KB@total,r3.06KB/w3.25KB@cta,r144B/w104B@thread;smem:r1.50KB/w1.50KB@logical,r1.50KB/w1.50KB@total,r1.50KB/w1.50KB@cta,r48B/w1.50KB@thread footprint=a:512B;b:256B footprint-precision=lower_bound peak=gmem:1.50KB;rmem:1.25KB;smem:1.50KB persistent=gmem:1.50KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(a: Tensor[(16, 32), "bf16"], b: Tensor[(32, 8), "bf16"], out: Tensor[(16, 8), "bf16"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        acc = T.alloc_tensor(
            tensor_type=Tensor[(16, 8), "f32", ((2, 4, 8, 2), (1, 2, 8, 64)), "rmem"]
        )
        lhs_frag = T.alloc_tensor(
            tensor_type=Tensor[(16, 16), "bf16", ((2, 4, 2, 8, 2), (1, 2, 8, 16, 128)), "rmem"]
        )
        rhs_frag = T.alloc_tensor(
            tensor_type=Tensor[(16, 8), "bf16", ((8, 2, 4, 2), (1, 8, 16, 64)), "rmem"]
        )
        result = T.alloc_tensor(
            tensor_type=Tensor[(16, 8), "bf16", ((2, 4, 8, 2), (1, 2, 8, 64)), "rmem"]
        )
        with Mesh((Topology("thread", 64),), layout=(2, 32), names=("d0", "d1")) as scope:
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as threads:
                T.fill(acc, 0.0)
            lhs_stages = (
                T.tensor_view(512, dtype="bf16", storage="smem", layout=((16, 16), (16, 1)), shape=(16, 16)),
                T.tensor_view(1024, dtype="bf16", storage="smem", layout=((16, 16), (16, 1)), shape=(16, 16)),
            )
            rhs_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=((16, 8), (8, 1)), shape=(16, 8)),
                T.tensor_view(256, dtype="bf16", storage="smem", layout=((16, 8), (8, 1)), shape=(16, 8)),
            )
            for k in range(0, 32, 16):
                with scope[:1] as scope_1:
                    tile = T.tensor_view(
                        T.ptr_of(a[0:0 + 16, k:k + 16]),
                        layout=((16, 16), (32, 1)),
                        shape=(16, 16),
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_1:
                        T.copy_async_tensor(tile, lhs_stages[(k // 16) % 2])
                    tile_1 = T.tensor_view(
                        T.ptr_of(b[k:k + 16, 0:0 + 8]), layout=((16, 8), (8, 1)), shape=(16, 8)
                    )
                    with Mesh(scope_1, layout=(32,), names=("d0",)) as threads_2:
                        T.copy_async_tensor(tile_1, rhs_stages[(k // 16) % 2])
                with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_2:
                    with scope_2 as threads_3:
                        dst_frame = T.tensor_view(
                            T.ptr_of(lhs_frag[0:0 + 16, 0:0 + 16]),
                            layout=((2, 4 @ threads_3.d0, 2, 8 @ threads_3.d1, 2), (1, 2, 8, 16, 128)),
                            shape=(16, 16),
                        )
                        T.ldmatrix(lhs_stages[(k // 16) % 2], dst_frame)
                        dst_frame_1 = T.tensor_view(
                            T.ptr_of(rhs_frag[0:0 + 16, 0:0 + 8]),
                            layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2), (1, 8, 16, 64)),
                            shape=(16, 8),
                        )
                        T.copy(rhs_stages[(k // 16) % 2], dst_frame_1)
                        for o_m in range(0, 16, 16):
                            for o_n in range(0, 8, 8):
                                for o_k in range(0, 16, 16):
                                    acc_view = T.tensor_view(
                                        T.ptr_of(acc[o_m:o_m + 16, o_n:o_n + 8]),
                                        layout=((2, 4 @ threads_3.d0, 8 @ threads_3.d1, 2), (1, 2, 8, 64)),
                                        shape=(16, 8),
                                    )
                                    lhs_view = T.tensor_view(
                                        T.ptr_of(lhs_frag[o_m:o_m + 16, o_k:o_k + 16]),
                                        layout=((2, 4 @ threads_3.d0, 2, 8 @ threads_3.d1, 2), (1, 2, 8, 16, 128)),
                                        shape=(16, 16),
                                    )
                                    rhs_view = T.tensor_view(
                                        T.ptr_of(rhs_frag[o_k:o_k + 16, o_n:o_n + 8]),
                                        layout=((8 @ threads_3.d1, 2, 4 @ threads_3.d0, 2), (1, 8, 16, 64)),
                                        shape=(16, 8),
                                    )
                                    T.tiled_mma(
                                        acc_view,
                                        lhs_view,
                                        rhs_view,
                                        atom=T.cuda.sm80.Mma(mesh=threads_3),
                                    )
            with Mesh(scope[1:], layout=((4, 8), (1, 4)), names=("d0", "d1")) as scope_3:
                with scope_3 as threads_6:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 16, 0:0 + 8]),
                        layout=((2, 4 @ threads_6.d0, 8 @ threads_6.d1, 2), (1, 2, 8, 64)),
                        shape=(16, 8),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(result[0:0 + 16, 0:0 + 8]),
                        layout=((2, 4 @ threads_6.d0, 8 @ threads_6.d1, 2), (1, 2, 8, 64)),
                        shape=(16, 8),
                    )
                    T.cast(src_frame, dst_frame_2, dtype="bf16")
        with Mesh(
            (Topology("thread", 64),), layout=((4, 8), (1, 4)) + 32, names=("d0", "d1")
        ) as threads_7:
            result_view = T.tensor_view(
                T.ptr_of(result[0:0 + 16, 0:0 + 8]),
                layout=((2, 4 @ threads_7.d0, 8 @ threads_7.d1, 2), (1, 2, 8, 64)),
                shape=(16, 8),
            )
            T.copy(result_view, out)
