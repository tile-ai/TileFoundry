# analysis target=nvidia.h200_sxm module=Prep function=gate_g topology=cta wave=132/1280
# selection requested=memory executed=memory
# memory traffic=gmem:r6.19KB/w204.00KB@logical,r7.73MB/w255.00MB@total,r6.19KB/w204.00KB@cta,r60B/w192.09KB@thread;rmem:r211.80KB/w139.70KB@logical,r264.75MB/w174.62MB@total,r211.80KB/w139.70KB@cta,r1.81KB/w1.15KB@thread footprint=a:12.00KB;a_log:96B;dt_bias:96B;v32:212:24.00KB footprint-precision=exact peak=gmem:288.19KB;rmem:48.95KB persistent=gmem:96.19KB
#   buffer=a holds=36.19KB time=none space=cta.cb reuse=948.00KB fits=yes precision=exact
#   buffer=a_log holds=36.19KB time=none space=cta.bt,cta.cb reuse=12.28KB fits=yes precision=exact
#   buffer=dt_bias holds=36.19KB time=none space=cta.bt,cta.cb reuse=12.28KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gate_g(
    a: Tensor[(1, 1024, 48), "bf16", ((1, 1024, 48), (17039360, 16640, 1))], a_log: ConstTensor[(48,), "bf16"], dt_bias: ConstTensor[(48,), "bf16"], out: Tensor[(128, 8, 16, 3), "f32"]
):
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("bt", "cb")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("s", "g")) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3))
            T.fill(value, 0.0)
            for _ in range(cta.cb, 1, 80):
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                value_2 = T.tensor_view(T.ptr_of(a_log), layout=(48,), shape=(48,))
                value_3 = T.tensor_view(
                    T.ptr_of(value_2), layout=((1, 1, 16, 3), (0, 0, 3, 1)), shape=(1, 1, 16, 3)
                )
                value_4 = T.tensor_view(
                    T.ptr_of(value_1), layout=(1, 1, 16 @ scope.g, 3), shape=(1, 1, 16, 3)
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                    layout=(1, 1, 16 @ scope.g, 3),
                    shape=(1, 1, 16, 3),
                )
                T.copy(value_3, dst_view)
                al = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                src_view = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                    layout=(1, 1, 16 @ scope.g, 3),
                    shape=(1, 1, 16, 3),
                )
                T.cast(src_view, al, dtype="f32")
                value_5 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                T.unary(al, value_5, kind=UnaryKind.EXP)
                value_6 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                T.unary(value_5, value_6, kind=UnaryKind.NEG)
                value_7 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                value_8 = T.tensor_view(
                    T.ptr_of(a),
                    layout=((1, 1024, 48), (17039360, 16640, 1)),
                    shape=(1, 1024, 48),
                )
                value_9 = T.tensor_view(
                    T.ptr_of(value_8),
                    layout=((128, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(128, 8, 16, 3),
                )
                value_10 = T.tensor_view(
                    T.ptr_of(value_9),
                    layout=((128, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(128, 8, 16, 3),
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value_10[cta.bt * (64 // 8):cta.bt * (64 // 8) + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((8, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(8, 8, 16, 3),
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_7),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                dst_view_1 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_11, dst_view_1)
                aa = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_12[0:0 + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8 @ scope.s, 8, 16 @ scope.g, 3),
                    shape=(8, 8, 16, 3),
                )
                T.cast(src_view_1, aa, dtype="f32")
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                value_14 = T.tensor_view(T.ptr_of(dt_bias), layout=(48,), shape=(48,))
                value_15 = T.tensor_view(
                    T.ptr_of(value_14),
                    layout=((1, 1, 16, 3), (0, 0, 3, 1)),
                    shape=(1, 1, 16, 3),
                )
                value_16 = T.tensor_view(
                    T.ptr_of(value_13), layout=(1, 1, 16 @ scope.g, 3), shape=(1, 1, 16, 3)
                )
                dst_view_2 = T.tensor_view(
                    T.ptr_of(value_16[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                    layout=(1, 1, 16 @ scope.g, 3),
                    shape=(1, 1, 16, 3),
                )
                T.copy(value_15, dst_view_2)
                db = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.g, 3), "rmem"]
                )
                src_view_2 = T.tensor_view(
                    T.ptr_of(value_16[0:0 + 1, 0:0 + 1, 0:0 + 16, 0:0 + 3]),
                    layout=(1, 1, 16 @ scope.g, 3),
                    shape=(1, 1, 16, 3),
                )
                T.cast(src_view_2, db, dtype="f32")
                x = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.binary(aa, db, x, kind=BinaryKind.ADD)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bool", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 20.0)
                T.binary(x, scalar, value_17, kind=BinaryKind.GT)
                u = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.unary(x, u, kind=UnaryKind.EXP)
                v1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(u, scalar_1, v1, kind=BinaryKind.ADD)
                value_18 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bool", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1.0)
                T.binary(v1, scalar_2, value_18, kind=BinaryKind.EQ)
                value_19 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.unary(v1, value_19, kind=UnaryKind.LOG)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.binary(value_19, u, value_20, kind=BinaryKind.MUL)
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 1.0)
                T.binary(v1, scalar_3, value_21, kind=BinaryKind.SUB)
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.binary(value_20, value_21, value_22, kind=BinaryKind.DIV)
                lp = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.where(value_18, u, value_22, lp)
                sp = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.where(value_17, x, lp, sp)
                value_23 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.s, 8, 16 @ scope.g, 3), "rmem"
                    ]
                )
                T.binary(value_6, sp, value_23, kind=BinaryKind.MUL)
                value_24 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                value_25 = T.tensor_view(
                    T.ptr_of(value_24), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                value_26 = T.tensor_view(
                    T.ptr_of(value_25), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                window = T.tensor_view(
                    T.ptr_of(value_26[cta.bt * (64 // 8):cta.bt * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_23, window)
