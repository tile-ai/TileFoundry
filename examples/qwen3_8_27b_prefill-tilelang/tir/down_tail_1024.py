# analysis target=nvidia.h200_sxm module=ResidualTail function=tail topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r144.00KB/w16.00KB@logical,r18.00MB/w2.00MB@total,r144.00KB/w16.00KB@cta,r576B/w64B@thread;rmem:r272.23KB/w272.06KB@logical,r34.03MB/w34.01MB@total,r272.23KB/w272.06KB@cta,r1.30KB/w1.12KB@thread footprint=hidden:16.00KB;v31:212:16.00KB footprint-precision=lower_bound peak=gmem:46.00MB;rmem:32.03KB persistent=gmem:36.00MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tail(
    out: Tensor[(1024, 5120), "bf16"], ws: Tensor[(16384, 256), "f32"], hidden: Tensor[(1, 1024, 5120), "bf16"], out_1: Tensor[(1024, 5120), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(4, 8, 4), names=("d0", "d1", "d2")) as cta:
        y = T.alloc_tensor(tensor_type=Tensor[(32, 256), "bf16", (32, 8, 8, 4), "rmem"])
        value = T.alloc_tensor(tensor_type=Tensor[(32, 256), "bf16", (32, 8, 8, 4), "rmem"])
        value_1 = T.alloc_tensor(tensor_type=Tensor[(32, 256), "bf16", (32, 8, 8, 4), "rmem"])
        acc = T.alloc_tensor(tensor_type=Tensor[(32, 256), "f32", (32, 8, 8, 4), "rmem"])
        acc_1 = T.alloc_tensor(tensor_type=Tensor[(32, 256), "f32", (32, 8, 8, 4), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(32, 256), "f32", (32, 8, 8, 4), "rmem"])
        with Mesh((Topology("thread", 384),), layout=(3, 128), names=("d0", "d1")) as scope:
            with Mesh(scope[:2], layout=(32, 8), names=("d0", "d1")) as scope_1:
                tile = T.tensor_view(
                    T.ptr_of(hidden[0:0 + 1, 0:0 + 1024, 0:0 + 5120]),
                    layout=(1024, 5120),
                    shape=(1024, 5120),
                )
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 32, (16 + cta.d0) * 256:(16 + cta.d0) * 256 + 256]),
                    layout=((32, 256), (5120, 1)),
                    shape=(32, 256),
                )
                with scope_1 as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(value[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads.d0, 8, 8 @ threads.d1, 4),
                        shape=(32, 256),
                    )
                    T.copy(tile_1, dst_frame)
                tile_2 = T.tensor_view(
                    T.ptr_of(ws[((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32):((((cta.d0 * 8) + cta.d1) * 4) * 128) + (cta.d2 * 32) + 32, 0:0 + 256]),
                    layout=(32, 256),
                    shape=(32, 256),
                )
                with scope_1 as threads_1:
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_1.d0, 8, 8 @ threads_1.d1, 4),
                        shape=(32, 256),
                    )
                    T.copy(tile_2, dst_frame_1)
                for j in range(1, 4, 1):
                    tile_3 = T.tensor_view(
                        T.ptr_of(ws[(((((cta.d0 * 8) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32):(((((cta.d0 * 8) + cta.d1) * 4) + j) * 128) + (cta.d2 * 32) + 32, 0:0 + 256]),
                        layout=(32, 256),
                        shape=(32, 256),
                    )
                    with scope_1 as threads_2:
                        dst_frame_2 = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 32, 0:0 + 256]),
                            layout=(32 @ threads_2.d0, 8, 8 @ threads_2.d1, 4),
                            shape=(32, 256),
                        )
                        T.copy(tile_3, dst_frame_2)
                        lhs_frame = T.tensor_view(
                            T.ptr_of(acc[0:0 + 32, 0:0 + 256]),
                            layout=(32 @ threads_2.d0, 8, 8 @ threads_2.d1, 4),
                            shape=(32, 256),
                        )
                        rhs_frame = T.tensor_view(
                            T.ptr_of(value_2[0:0 + 32, 0:0 + 256]),
                            layout=(32 @ threads_2.d0, 8, 8 @ threads_2.d1, 4),
                            shape=(32, 256),
                        )
                        dst_frame_3 = T.tensor_view(
                            T.ptr_of(acc_1[0:0 + 32, 0:0 + 256]),
                            layout=(32 @ threads_2.d0, 8, 8 @ threads_2.d1, 4),
                            shape=(32, 256),
                        )
                        T.binary(lhs_frame, rhs_frame, dst_frame_3, kind=BinaryKind.ADD)
                        acc_1_view = T.tensor_view(
                            T.ptr_of(acc_1[0:0 + 32, 0:0 + 256]),
                            layout=(32 @ threads_2.d0, 8, 8 @ threads_2.d1, 4),
                            shape=(32, 256),
                        )
                        T.copy(acc_1_view, acc)
                with scope_1 as threads_5:
                    src_frame = T.tensor_view(
                        T.ptr_of(acc[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    T.cast(src_frame, dst_frame_4, dtype="bf16")
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(value[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(y[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    T.binary(lhs_frame_1, rhs_frame_1, dst_frame_5, kind=BinaryKind.ADD)
                    y_view = T.tensor_view(
                        T.ptr_of(y[0:0 + 32, 0:0 + 256]),
                        layout=(32 @ threads_5.d0, 8, 8 @ threads_5.d1, 4),
                        shape=(32, 256),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32):(((0 * 8) + cta.d1) * 128) + (cta.d2 * 32) + 32, (16 + cta.d0) * 256:(16 + cta.d0) * 256 + 256]),
                        layout=((32, 256), (5120, 1)),
                        shape=(32, 256),
                    )
                    T.copy(y_view, window)
        T.copy(out, out_1)
