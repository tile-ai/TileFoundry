# analysis target=nvidia.h200_sxm module=Prep function=gate_beta topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r6.00KB/w102.00KB@logical,r7.50MB/w127.50MB@total,r6.00KB/w102.00KB@cta,r48B/w96.05KB@thread;rmem:r72.10KB/w72.01KB@logical,r90.13MB/w90.01MB@total,r72.10KB/w72.01KB@cta,r680B/w584B@thread footprint=b:6.00KB;v14:224:6.00KB footprint-precision=exact peak=gmem:192.00KB;rmem:18.01KB persistent=gmem:96.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gate_beta(
    b: Tensor[(1, 1024, 48), "bf16", ((1, 1024, 48), (17039360, 16640, 1))], out: Tensor[(128, 8, 16, 3), "bf16"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        value = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "bf16", (8, 8, 16, 3), "rmem"])
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        bb = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "bf16", (8, 8, 16, 3), "rmem"]
        )
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(b[0:0 + 1, 0:0 + 1024, 0:0 + 48]),
                layout=(128, 8, 16, 3),
                shape=(128, 8, 16, 3),
            )
            for _ in range(cta.d1, 1, 80):
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 1.0)
                tile_1 = T.tensor_view(
                    T.ptr_of(tile[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                with scope as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.copy(tile_1, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(bb[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.cast(src_frame, dst_frame_1, dtype="f32")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(bb[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.unary(src_frame_1, dst_frame_2, kind=UnaryKind.NEG)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads.d0, 8, 16 @ threads.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.unary(src_frame_2, dst_frame_3, kind=UnaryKind.EXP)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                with scope as threads_4:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame, scalar_1, dst_frame_4, kind=BinaryKind.ADD)
                    rhs_frame = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(scalar, rhs_frame, dst_frame_5, kind=BinaryKind.DIV)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.cast(src_frame_3, dst_frame_6, dtype="bf16")
                    value_view = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8, 8, 16, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.copy(value_view, window)
