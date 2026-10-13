# analysis target=nvidia.h200_sxm module=Prep function=gate_g topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r6.19KB/w204.00KB@logical,r7.73MB/w255.00MB@total,r6.19KB/w204.00KB@cta,r60B/w192.09KB@thread;rmem:r211.80KB/w139.70KB@logical,r264.75MB/w174.62MB@total,r211.80KB/w139.70KB@cta,r1.81KB/w1.15KB@thread footprint=a:6.00KB;a_log:96B;dt_bias:96B;v29:244:12.00KB footprint-precision=exact peak=gmem:288.19KB;rmem:18.20KB persistent=gmem:96.19KB
#   buffer=a_log holds=18.19KB time=none space=cta.bt,cta.cb reuse=12.28KB fits=yes precision=exact
#   buffer=dt_bias holds=18.19KB time=none space=cta.bt,cta.cb reuse=12.28KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gate_g(
    a: Tensor[(1, 1024, 48), "bf16", ((1, 1024, 48), (17039360, 16640, 1))], a_log: Tensor[(48,), "bf16"], dt_bias: Tensor[(48,), "bf16"], out: Tensor[(128, 8, 16, 3), "f32"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        T.fill(out, 0.0)
        value = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_1 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16, 3), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16, 3), "rmem"]
        )
        al = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16, 3), "rmem"])
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16, 3), "rmem"]
        )
        sp = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "bool", (8, 8, 16, 3), "rmem"]
        )
        x = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        aa = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "bf16", (8, 8, 16, 3), "rmem"]
        )
        db = T.alloc_tensor(tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16, 3), "rmem"])
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16, 3), "rmem"]
        )
        lp = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "bool", (8, 8, 16, 3), "rmem"]
        )
        v1 = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        u = T.alloc_tensor(tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"])
        value_8 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_9 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_10 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        value_11 = T.alloc_tensor(
            tensor_type=Tensor[(8, 8, 16, 3), "f32", (8, 8, 16, 3), "rmem"]
        )
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(a[0:0 + 1, 0:0 + 1024, 0:0 + 48]),
                layout=(128, 8, 16, 3),
                shape=(128, 8, 16, 3),
            )
            for _ in range(cta.d1, 1, 80):
                tile_1 = T.tensor_view(
                    T.ptr_of(a_log[0:0 + 48]), layout=(1, 1, 16, 3), shape=(1, 1, 16, 3)
                )
                with scope as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.copy(tile_1, dst_frame)
                    src_frame = T.tensor_view(
                        T.ptr_of(value_3[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    dst_frame_1 = T.tensor_view(
                        T.ptr_of(al[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.cast(src_frame, dst_frame_1, dtype="f32")
                    src_frame_1 = T.tensor_view(
                        T.ptr_of(al[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    dst_frame_2 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.unary(src_frame_1, dst_frame_2, kind=UnaryKind.EXP)
                    src_frame_2 = T.tensor_view(
                        T.ptr_of(value_2[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    dst_frame_3 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.unary(src_frame_2, dst_frame_3, kind=UnaryKind.NEG)
                tile_2 = T.tensor_view(
                    T.ptr_of(tile[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                with scope as threads_4:
                    dst_frame_4 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.copy(tile_2, dst_frame_4)
                    src_frame_3 = T.tensor_view(
                        T.ptr_of(value_5[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_5 = T.tensor_view(
                        T.ptr_of(aa[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_4.d0, 8, 16 @ threads_4.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.cast(src_frame_3, dst_frame_5, dtype="f32")
                tile_3 = T.tensor_view(
                    T.ptr_of(dt_bias[0:0 + 48]), layout=(1, 1, 16, 3), shape=(1, 1, 16, 3)
                )
                with scope as threads_6:
                    dst_frame_6 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads_6.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.copy(tile_3, dst_frame_6)
                    src_frame_4 = T.tensor_view(
                        T.ptr_of(value_6[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads_6.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    dst_frame_7 = T.tensor_view(
                        T.ptr_of(db[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads_6.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    T.cast(src_frame_4, dst_frame_7, dtype="f32")
                    lhs_frame = T.tensor_view(
                        T.ptr_of(aa[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_6.d0, 8, 16 @ threads_6.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    rhs_frame = T.tensor_view(
                        T.ptr_of(db[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads_6.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    dst_frame_8 = T.tensor_view(
                        T.ptr_of(x[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_6.d0, 8, 16 @ threads_6.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame, rhs_frame, dst_frame_8, kind=BinaryKind.ADD)
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 20.0)
                with scope as threads_9:
                    lhs_frame_1 = T.tensor_view(
                        T.ptr_of(x[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_9.d0, 8, 16 @ threads_9.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_9 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_9.d0, 8, 16 @ threads_9.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_1, scalar, dst_frame_9, kind=BinaryKind.GT)
                    src_frame_5 = T.tensor_view(
                        T.ptr_of(x[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_9.d0, 8, 16 @ threads_9.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_10 = T.tensor_view(
                        T.ptr_of(u[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_9.d0, 8, 16 @ threads_9.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.unary(src_frame_5, dst_frame_10, kind=UnaryKind.EXP)
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                with scope as threads_11:
                    lhs_frame_2 = T.tensor_view(
                        T.ptr_of(u[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_11.d0, 8, 16 @ threads_11.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_11 = T.tensor_view(
                        T.ptr_of(v1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_11.d0, 8, 16 @ threads_11.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_2, scalar_1, dst_frame_11, kind=BinaryKind.ADD)
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1.0)
                with scope as threads_12:
                    lhs_frame_3 = T.tensor_view(
                        T.ptr_of(v1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_12 = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_3, scalar_2, dst_frame_12, kind=BinaryKind.EQ)
                    src_frame_6 = T.tensor_view(
                        T.ptr_of(v1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_13 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.unary(src_frame_6, dst_frame_13, kind=UnaryKind.LOG)
                    lhs_frame_4 = T.tensor_view(
                        T.ptr_of(value_10[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    rhs_frame_1 = T.tensor_view(
                        T.ptr_of(u[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_14 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_12.d0, 8, 16 @ threads_12.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_4, rhs_frame_1, dst_frame_14, kind=BinaryKind.MUL)
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 1.0)
                with scope as threads_15:
                    lhs_frame_5 = T.tensor_view(
                        T.ptr_of(v1[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_15 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_5, scalar_3, dst_frame_15, kind=BinaryKind.SUB)
                    lhs_frame_6 = T.tensor_view(
                        T.ptr_of(value_9[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    rhs_frame_2 = T.tensor_view(
                        T.ptr_of(value_11[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_16 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_6, rhs_frame_2, dst_frame_16, kind=BinaryKind.DIV)
                    cond_frame = T.tensor_view(
                        T.ptr_of(value_7[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    lhs_frame_7 = T.tensor_view(
                        T.ptr_of(u[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    rhs_frame_3 = T.tensor_view(
                        T.ptr_of(value_8[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_17 = T.tensor_view(
                        T.ptr_of(lp[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.where(cond_frame, lhs_frame_7, rhs_frame_3, dst_frame_17)
                    cond_frame_1 = T.tensor_view(
                        T.ptr_of(value_4[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    lhs_frame_8 = T.tensor_view(
                        T.ptr_of(x[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    rhs_frame_4 = T.tensor_view(
                        T.ptr_of(lp[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_18 = T.tensor_view(
                        T.ptr_of(sp[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.where(cond_frame_1, lhs_frame_8, rhs_frame_4, dst_frame_18)
                    lhs_frame_9 = T.tensor_view(
                        T.ptr_of(value_1[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                        layout=(1, 1, 16 @ threads_15.d1, 3),
                        shape=(1, 1, 16, 3),
                    )
                    rhs_frame_5 = T.tensor_view(
                        T.ptr_of(sp[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    dst_frame_19 = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.binary(lhs_frame_9, rhs_frame_5, dst_frame_19, kind=BinaryKind.MUL)
                    value_view = T.tensor_view(
                        T.ptr_of(value[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8 @ threads_15.d0, 8, 16 @ threads_15.d1, 3),
                        shape=(8, 8, 16, 3),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                        layout=(8, 8, 16, 3),
                        shape=(8, 8, 16, 3),
                    )
                    T.copy(value_view, window)
