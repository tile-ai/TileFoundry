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
    with Mesh((Topology("cta", 1280),), layout=(16, 80), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(8, 16), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            for _ in range(cta.d1, 1, 80):
                value = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                value_1 = T.tensor_view(
                    T.ptr_of(a_log), layout=((1, 1, 16, 3), (0, 0, 3, 1)), shape=(1, 1, 16, 3)
                )
                value_2 = T.tensor_view(
                    T.ptr_of(value), layout=(1, 1, 16 @ scope.d1, 3), shape=(1, 1, 16, 3)
                )
                T.copy(value_1, value_2)
                al = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                T.cast(value_2, al, dtype="f32")
                value_3 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                T.unary(al, value_3, kind=UnaryKind.EXP)
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                T.unary(value_3, value_4, kind=UnaryKind.NEG)
                value_5 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bf16", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                value_6 = T.tensor_view(
                    T.ptr_of(a[0:0 + 1, 8 * (cta.d0 * (64 // 8)):8 * (cta.d0 * (64 // 8)) + 1, 0:0 + 1]),
                    layout=((8, 8, 16, 3), (133120, 16640, 3, 1)),
                    shape=(8, 8, 16, 3),
                )
                value_7 = T.tensor_view(
                    T.ptr_of(value_5),
                    layout=(8 @ scope.d0, 8, 16 @ scope.d1, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_6, value_7)
                aa = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.cast(value_7, aa, dtype="f32")
                value_8 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "bf16", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                value_9 = T.tensor_view(
                    T.ptr_of(dt_bias), layout=((1, 1, 16, 3), (0, 0, 3, 1)), shape=(1, 1, 16, 3)
                )
                value_10 = T.tensor_view(
                    T.ptr_of(value_8), layout=(1, 1, 16 @ scope.d1, 3), shape=(1, 1, 16, 3)
                )
                T.copy(value_9, value_10)
                db = T.alloc_tensor(
                    tensor_type=Tensor[(1, 1, 16, 3), "f32", (1, 1, 16 @ scope.d1, 3), "rmem"]
                )
                T.cast(value_10, db, dtype="f32")
                x = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.binary(aa, db, x, kind=BinaryKind.ADD)
                value_11 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bool", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar, 20.0)
                T.binary(x, scalar, value_11, kind=BinaryKind.GT)
                u = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.unary(x, u, kind=UnaryKind.EXP)
                v1 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_1, 1.0)
                T.binary(u, scalar_1, v1, kind=BinaryKind.ADD)
                value_12 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "bool", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar_2 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_2, 1.0)
                T.binary(v1, scalar_2, value_12, kind=BinaryKind.EQ)
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.unary(v1, value_13, kind=UnaryKind.LOG)
                value_14 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.binary(value_13, u, value_14, kind=BinaryKind.MUL)
                value_15 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                scalar_3 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
                T.fill(scalar_3, 1.0)
                T.binary(v1, scalar_3, value_15, kind=BinaryKind.SUB)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.binary(value_14, value_15, value_16, kind=BinaryKind.DIV)
                lp = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.where(value_12, u, value_16, lp)
                sp = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.where(value_11, x, lp, sp)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (8, 8, 16, 3), "f32", (8 @ scope.d0, 8, 16 @ scope.d1, 3), "rmem"
                    ]
                )
                T.binary(value_4, sp, value_17, kind=BinaryKind.MUL)
                value_18 = T.tensor_view(
                    T.ptr_of(out), layout=(128, 8, 16, 3), shape=(128, 8, 16, 3)
                )
                window = T.tensor_view(
                    T.ptr_of(value_18[cta.d0 * (64 // 8):cta.d0 * (64 // 8) + 8, 0:0 + 8, 0:0 + 16, 0:0 + 3]),
                    layout=(8, 8, 16, 3),
                    shape=(8, 8, 16, 3),
                )
                T.copy(value_17, window)
