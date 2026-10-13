# analysis target=nvidia.h200_sxm module=Prep function=conv_v topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r17.75KB/w12.02MB@logical,r22.19MB/w15.02GB@total,r17.75KB/w12.02MB@cta,r16.11KB/w12.00MB@thread;rmem:r645.01KB/w612.56KB@logical,r806.26MB/w765.71MB@total,r645.01KB/w612.56KB@cta,r5.79KB/w5.10KB@thread;smem:r80.00KB/w49.50KB@logical,r100.00MB/w61.88MB@total,r80.00KB/w49.50KB@cta,r16.50KB/w48.80KB@thread footprint=conv_w:48.00KB;v82:212:768.00KB footprint-precision=lower_bound peak=gmem:32.08MB;rmem:48.02KB;smem:16.75KB persistent=gmem:20.08MB
#   buffer=conv_w holds=816.00KB time=none space=cta.bt reuse=48.00KB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def conv_v(
    qkv: Tensor[(1, 1024, 10240), "bf16", ((1, 1024, 10240), (17039360, 16640, 1))], conv_w: Tensor[(10240, 1, 4), "bf16"], out: Tensor[(128, 8, 48, 16, 8), "bf16"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        y = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"])
        value = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        xb = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"]
        )
        acc = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        acc_1 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        acc_2 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        acc_3 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"])
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        x0 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"])
        slab = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(67, 16, 8), shape=(67, 16, 8)
        )
        value_3 = T.alloc_tensor(tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16, 8), "rmem"])
        hist = T.alloc_tensor(tensor_type=Tensor[(3, 16, 8), "bf16", (3, 16, 8), "rmem"])
        keep = T.alloc_tensor(tensor_type=Tensor[(), "bf16", "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16, 8), "rmem"]
        )
        w0 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16, 8), "rmem"])
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        x1 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"])
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16, 8), "rmem"]
        )
        w1 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16, 8), "rmem"])
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        x2 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"])
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16, 8), "rmem"]
        )
        w2_ = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16, 8), "rmem"])
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        value_12 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        x3 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 8), "bf16", (8, 8, 16, 8), "rmem"])
        value_13 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 8), "f32", (1, 1, 16, 8), "rmem"]
        )
        w3 = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 8), "bf16", (1, 1, 16, 8), "rmem"])
        value_14 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        value_15 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        value_16 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 8), "f32", (8, 8, 16, 8), "rmem"]
        )
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(qkv[0:0 + 1, 0:0 + 1024, 0:0 + 10240]),
                layout=((1024, 10240), (16640, 1)),
                shape=(1024, 10240),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(conv_w[0:0 + 10240, 0:0 + 1, 0:0 + 4]),
                layout=(10240, 4),
                shape=(10240, 4),
            )
            for h in range((cta.d1 - (80 * ((cta.d1 - 32) // 80))) - 32, 48, 80):
                T.fill(slab, 0.0)
                tile_2 = T.tensor_view(
                    T.ptr_of(tile[(((cta.d0 * 64) + 1024) - (4 - 1)) % 1024:(((cta.d0 * 64) + 1024) - (4 - 1)) % 1024 + 3, ((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128]),
                    layout=((3, 128), (16640, 1)),
                    shape=(3, 128),
                )
                tile_3 = T.tensor_view(
                    T.ptr_of(tile_2[0:0 + 3, 0:0 + 128]), layout=(3, 16, 8), shape=(3, 16, 8)
                )
                with scope as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(hist[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                        layout=(3, 16 @ threads.d1, 8),
                        shape=(3, 16, 8),
                    )
                    T.copy(tile_3, dst_frame)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "i64", (), "rmem"])
                T.fill(scalar, ceildiv(cta.d0, 16))
                with scope[:] as threads_1:
                    T.cast(scalar, keep, dtype="bf16")
                with scope as threads_2:
                    lhs_frame = T.tensor_view(
                        T.ptr_of(hist[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                        layout=(3, 16 @ threads_2.d1, 8),
                        shape=(3, 16, 8),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                        layout=(3, 16 @ threads_2.d1, 8),
                        shape=(3, 16, 8),
                    )
                    T.binary(lhs_frame, keep, dst_frame_1, kind=BinaryKind.MUL)
                    value_3_view = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                        layout=(3, 16 @ threads_2.d1, 8),
                        shape=(3, 16, 8),
                    )
                    window = T.tensor_view(
                        T.ptr_of(slab[0:0 + 3, 0:0 + 16, 0:0 + 8]),
                        layout=(3, 16, 8),
                        shape=(3, 16, 8),
                    )
                    T.copy(value_3_view, window)
                tile_4 = T.tensor_view(
                    T.ptr_of(tile[cta.d0 * 64:cta.d0 * 64 + 64, ((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128]),
                    layout=((64, 128), (16640, 1)),
                    shape=(64, 128),
                )
                tile_5 = T.tensor_view(
                    T.ptr_of(tile_4[0:0 + 64, 0:0 + 128]), layout=(64, 16, 8), shape=(64, 16, 8)
                )
                with scope[:] as threads_4:
                    window_1 = T.tensor_view(
                        T.ptr_of(slab[3:3 + 64, 0:0 + 16, 0:0 + 8]),
                        layout=(64, 16, 8) + 384,
                        shape=(64, 16, 8),
                    )
                    T.copy(tile_5, window_1)
                tile_6 = T.tensor_view(
                    T.ptr_of(slab[3:3 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8) + 384,
                    shape=(64, 16, 8),
                )
                tile_7 = T.tensor_view(
                    T.ptr_of(tile_6[0:0 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(8, 8, 16, 8) + 384,
                    shape=(8, 8, 16, 8),
                )
                with scope as threads_5:
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(x0[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_5.d0, 8, 16 @ threads_5.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.copy(tile_7, dst_frame_2)
                    src_frame = T.tensor_view(
                        T.ptr_of(x0[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_5.d0, 8, 16 @ threads_5.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_5.d0, 8, 16 @ threads_5.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame, dst_frame_3, dtype="f32")
                tile_8 = T.tensor_view(
                    T.ptr_of(tile_1[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128, 3:3 + 1]),
                    layout=((128, 1), (4, 1)),
                    shape=(128, 1),
                )
                tile_9 = T.tensor_view(
                    T.ptr_of(tile_8[0:0 + 128, 0:0 + 1]),
                    layout=(1, 1, 16, 8),
                    shape=(1, 1, 16, 8),
                )
                with scope as threads_7:
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(w0[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_7.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.copy(tile_9, dst_frame_4)
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(w0[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_7.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_7.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.cast(src_frame_1, dst_frame_5, dtype="f32")
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_7.d0, 8, 16 @ threads_7.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_7.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(acc_3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_7.d0, 8, 16 @ threads_7.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_1, rhs_frame, dst_frame_6, kind=BinaryKind.MUL)
                tile_10 = T.tensor_view(
                    T.ptr_of(slab[2:2 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8) + 256,
                    shape=(64, 16, 8),
                )
                tile_11 = T.tensor_view(
                    T.ptr_of(tile_10[0:0 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(8, 8, 16, 8) + 256,
                    shape=(8, 8, 16, 8),
                )
                with scope as threads_10:
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(x1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_10.d0, 8, 16 @ threads_10.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.copy(tile_11, dst_frame_7)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(x1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_10.d0, 8, 16 @ threads_10.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_10.d0, 8, 16 @ threads_10.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_2, dst_frame_8, dtype="f32")
                tile_12 = T.tensor_view(
                    T.ptr_of(tile_1[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128, 2:2 + 1]),
                    layout=((128, 1), (4, 1)),
                    shape=(128, 1),
                )
                tile_13 = T.tensor_view(
                    T.ptr_of(tile_12[0:0 + 128, 0:0 + 1]),
                    layout=(1, 1, 16, 8),
                    shape=(1, 1, 16, 8),
                )
                with scope as threads_12:
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(w1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_12.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.copy(tile_13, dst_frame_9)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(w1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_12.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_12.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.cast(src_frame_3, dst_frame_10, dtype="f32")
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_12.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_2, rhs_frame_1, dst_frame_11, kind=BinaryKind.MUL)
                    lhs_frame_3 = T.tensor_view(
                        T.ptr_of(acc_3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_12 = T.tensor_view(
                        T.ptr_of(acc_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_3, rhs_frame_2, dst_frame_12, kind=BinaryKind.ADD)
                tile_14 = T.tensor_view(
                    T.ptr_of(slab[1:1 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8) + 128,
                    shape=(64, 16, 8),
                )
                tile_15 = T.tensor_view(
                    T.ptr_of(tile_14[0:0 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(8, 8, 16, 8) + 128,
                    shape=(8, 8, 16, 8),
                )
                with scope as threads_16:
                    dst_frame_13 = T.tensor_view(
                        T.ptr_of(x2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_16.d0, 8, 16 @ threads_16.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.copy(tile_15, dst_frame_13)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(x2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_16.d0, 8, 16 @ threads_16.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_14 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_16.d0, 8, 16 @ threads_16.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_4, dst_frame_14, dtype="f32")
                tile_16 = T.tensor_view(
                    T.ptr_of(tile_1[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128, 1:1 + 1]),
                    layout=((128, 1), (4, 1)),
                    shape=(128, 1),
                )
                tile_17 = T.tensor_view(
                    T.ptr_of(tile_16[0:0 + 128, 0:0 + 1]),
                    layout=(1, 1, 16, 8),
                    shape=(1, 1, 16, 8),
                )
                with scope as threads_18:
                    dst_frame_15 = T.tensor_view(
                        T.ptr_of(w2_[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_18.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.copy(tile_17, dst_frame_15)
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(w2_[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_18.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_16 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_18.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.cast(src_frame_5, dst_frame_16, dtype="f32")
                    lhs_frame_4 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_18.d0, 8, 16 @ threads_18.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_18.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_17 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_18.d0, 8, 16 @ threads_18.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_4, rhs_frame_3, dst_frame_17, kind=BinaryKind.MUL)
                    lhs_frame_5 = T.tensor_view(
                        T.ptr_of(acc_2[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_18.d0, 8, 16 @ threads_18.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_4 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_18.d0, 8, 16 @ threads_18.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_18 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_18.d0, 8, 16 @ threads_18.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_5, rhs_frame_4, dst_frame_18, kind=BinaryKind.ADD)
                tile_18 = T.tensor_view(
                    T.ptr_of(slab[0:0 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(64, 16, 8),
                    shape=(64, 16, 8),
                )
                tile_19 = T.tensor_view(
                    T.ptr_of(tile_18[0:0 + 64, 0:0 + 16, 0:0 + 8]),
                    layout=(8, 8, 16, 8),
                    shape=(8, 8, 16, 8),
                )
                with scope as threads_22:
                    dst_frame_19 = T.tensor_view(
                        T.ptr_of(x3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_22.d0, 8, 16 @ threads_22.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.copy(tile_19, dst_frame_19)
                    src_frame_6 = T.tensor_view(
                        T.ptr_of(x3[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_22.d0, 8, 16 @ threads_22.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_20 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_22.d0, 8, 16 @ threads_22.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_6, dst_frame_20, dtype="f32")
                tile_20 = T.tensor_view(
                    T.ptr_of(tile_1[((2 * 16) + h) * 128:((2 * 16) + h) * 128 + 128, 0:0 + 1]),
                    layout=((128, 1), (4, 1)),
                    shape=(128, 1),
                )
                tile_21 = T.tensor_view(
                    T.ptr_of(tile_20[0:0 + 128, 0:0 + 1]),
                    layout=(1, 1, 16, 8),
                    shape=(1, 1, 16, 8),
                )
                with scope as threads_24:
                    dst_frame_21 = T.tensor_view(
                        T.ptr_of(w3[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_24.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.copy(tile_21, dst_frame_21)
                    src_frame_7 = T.tensor_view(
                        T.ptr_of(w3[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_24.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_22 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_24.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    T.cast(src_frame_7, dst_frame_22, dtype="f32")
                    lhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_12[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_5 = T.tensor_view(
                        T.ptr_of(value_13[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 8]),
                        layout=(1, 1, 16 @ threads_24.d1, 8),
                        shape=(1, 1, 16, 8),
                    )
                    dst_frame_23 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_6, rhs_frame_5, dst_frame_23, kind=BinaryKind.MUL)
                    lhs_frame_7 = T.tensor_view(
                        T.ptr_of(acc_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_24 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_7, rhs_frame_6, dst_frame_24, kind=BinaryKind.ADD)
                    src_frame_8 = T.tensor_view(
                        T.ptr_of(acc[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_25 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_8, dst_frame_25, dtype="bf16")
                    src_frame_9 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_26 = T.tensor_view(
                        T.ptr_of(xb[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_9, dst_frame_26, dtype="f32")
                    src_frame_10 = T.tensor_view(
                        T.ptr_of(xb[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_27 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.unary(src_frame_10, dst_frame_27, kind=UnaryKind.NEG)
                    src_frame_11 = T.tensor_view(
                        T.ptr_of(value_16[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_28 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_24.d0, 8, 16 @ threads_24.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.unary(src_frame_11, dst_frame_28, kind=UnaryKind.EXP)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                with scope as threads_32:
                    lhs_frame_8 = T.tensor_view(
                        T.ptr_of(value_15[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_29 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_8, scalar_1, dst_frame_29, kind=BinaryKind.ADD)
                    lhs_frame_9 = T.tensor_view(
                        T.ptr_of(xb[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    rhs_frame_7 = T.tensor_view(
                        T.ptr_of(value_14[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_30 = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.binary(lhs_frame_9, rhs_frame_7, dst_frame_30, kind=BinaryKind.DIV)
                    src_frame_12 = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    dst_frame_31 = T.tensor_view(
                        T.ptr_of(y[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                        layout=(8 @ threads_32.d0, 8, 16 @ threads_32.d1, 8),
                        shape=(8, 8, 16, 8),
                    )
                    T.cast(src_frame_12, dst_frame_31, dtype="bf16")
                tile_22 = T.tensor_view(
                    T.ptr_of(y[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 8]),
                    layout=(8, 8, 1, 16, 8),
                    shape=(8, 8, 1, 16, 8),
                )
                window_2 = T.tensor_view(
                    T.ptr_of(out[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, h:h + 1, 0:0 + 16, 0:0 + 8]),
                    layout=((8, 8, 1, 16, 8), (49152, 6144, 128, 8, 1)),
                    shape=(8, 8, 1, 16, 8),
                )
                T.copy(tile_22, window_2)
