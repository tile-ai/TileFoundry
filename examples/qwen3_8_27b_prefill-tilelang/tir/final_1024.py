# analysis target=nvidia.h200_sxm module=Final function=final topology=cta wave=132/7760
# selection requested=memory executed=memory
# memory traffic=gmem:r340.01KB/w485.06KB@logical,r2.52GB/w3.59GB@total,r340.01KB/w485.06KB@cta,r1.34KB/w485.01KB@thread;rmem:r2.40MB/w2.38MB@logical,r18.21GB/w18.06GB@total,r2.40MB/w2.38MB@cta,r10.63KB/w10.42KB@thread;smem:r10.00KB/w10.00KB@logical,r75.78MB/w75.78MB@total,r10.00KB/w10.00KB@cta,r320B/w40B@thread footprint=<value 0>[0]:8B;gamma_final:10.00KB;hidden:10.00KB;v46:86:8.25KB;w_head:41.25MB footprint-precision=upper_bound peak=gmem:2.38GB;rmem:1.25MB;smem:10.00KB persistent=gmem:2.38GB
#   buffer=hidden holds=41.28MB time=none space=cta.b reuse=1.28MB fits=yes precision=upper_bound
#   buffer=gamma_final holds=41.28MB time=none space=cta.b reuse=1.28MB fits=yes precision=upper_bound
#   buffer=<value 0>[0] holds=41.28MB time=none space=cta.b reuse=1.02KB fits=yes precision=upper_bound
#   error="rmem placement peak 1.25MB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def final(
    hidden: Tensor[(1, 1024, 5120), "bf16"], last: Tensor[(1,), "i64"], gamma_final: ConstTensor[(5120,), "bf16"], w_head: ConstTensor[(248320, 5120), "bf16"], out: Tensor[(62080, 4, 1, 1, 1), "bf16"]
):
    with Mesh((Topology("cta", 7760),), layout=(7760,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(256,), names=("d0",)) as scope:
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            value_2 = T.tensor_view(T.ptr_of(last), layout=(), shape=())
            value_3 = T.tensor_view(
                T.ptr_of(hidden[0:0 + 1, value_2:value_2 + 1, 0:0 + 1]),
                layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                shape=(1, 5, 256, 4),
            )
            value_4 = T.tensor_view(
                T.ptr_of(value_1), layout=(1, 5, 256 @ scope.d0, 4), shape=(1, 5, 256, 4)
            )
            T.copy(value_3, value_4)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_4, x32, dtype="f32")
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            value_6 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5, 256, 4), "f32", ((1, 5, 256 @ scope.d0, 4), (0, 4, 0, 1)), "rmem"
                ]
            )
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.unary(x32, value_7, kind=UnaryKind.SQUARE)
            value_8 = T.tensor_view(
                T.ptr_of(value_6),
                layout=((1, 5, 256 @ scope.d0, 4), (0, 4, 0, 1)),
                shape=(1, 5, 256, 4),
            )
            T.copy(value_7, value_8)
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 1, 1),
                    "f32",
                    ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)) @ {B()},
                    "rmem",
                ]
            )
            T.reduce(value_8, value_9, axes=(1, 2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_10 = T.tensor_view(T.ptr_of(value_5), layout=(1, 1, 1, 1), shape=(1, 1, 1, 1))
            T.copy(value_9, value_10)
            value_11 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(value_10, scalar, value_11, kind=BinaryKind.ADD)
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[(1, 1, 1, 1), "f32", (1, 1, 1, 1), "rmem"]
            )
            T.unary(value_11, value_12, kind=UnaryKind.RSQRT)
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.binary(x32, value_12, value_13, kind=BinaryKind.MUL)
            value_14 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            value_15 = T.tensor_view(
                T.ptr_of(gamma_final),
                layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                shape=(1, 5, 256, 4),
            )
            value_16 = T.tensor_view(
                T.ptr_of(value_14), layout=(1, 5, 256 @ scope.d0, 4), shape=(1, 5, 256, 4)
            )
            T.copy(value_15, value_16)
            value_17 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_16, value_17, dtype="f32")
            value_18 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_17, scalar_1, value_18, kind=BinaryKind.ADD)
            value_19 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.binary(value_13, value_18, value_19, kind=BinaryKind.MUL)
            xn = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_19, xn, dtype="bf16")
            value = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_20 = T.tensor_view(
                T.ptr_of(value), layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            T.copy(xn, value_20)
        with Mesh((Topology("thread", 256),), layout=(8, 32), names=("d0", "d1")) as scope_1:
            T.fill(out, 0.0)
            value_21 = T.alloc_tensor(
                tensor_type=Tensor[(8, 4, 1, 1, 1), "f32", (8 @ scope_1.d0, 4, 1, 1, 1), "rmem"]
            )
            value_22 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8),
                    "f32",
                    ((8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8), (0, 160, 8, 0, 1)),
                    "rmem",
                ]
            )
            value_23 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8),
                    "bf16",
                    (8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8),
                    "rmem",
                ]
            )
            value_24 = T.tensor_view(
                T.ptr_of(w_head[4 * ((cta.d0 * 32) // 4):4 * ((cta.d0 * 32) // 4) + 1, 0:0 + 1]),
                layout=(8, 4, 20, 32, 8),
                shape=(8, 4, 20, 32, 8),
            )
            value_25 = T.tensor_view(
                T.ptr_of(value_23),
                layout=(8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8),
                shape=(8, 4, 20, 32, 8),
            )
            T.copy(value_24, value_25)
            value_26 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8),
                    "f32",
                    (8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8),
                    "rmem",
                ]
            )
            T.cast(value_25, value_26, dtype="f32")
            value_27 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 20, 32, 8), "bf16", (1, 1, 20, 32 @ scope_1.d1, 8), "rmem"
                ]
            )
            value_36 = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(1, 5, 256, 4), shape=(1, 5, 256, 4)
            )
            value_28 = T.tensor_view(
                T.ptr_of(value_36),
                layout=((1, 1, 20, 32, 8), (0, 0, 256, 8, 1)),
                shape=(1, 1, 20, 32, 8),
            )
            value_29 = T.tensor_view(
                T.ptr_of(value_27),
                layout=(1, 1, 20, 32 @ scope_1.d1, 8),
                shape=(1, 1, 20, 32, 8),
            )
            T.copy(value_28, value_29)
            value_30 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 1, 20, 32, 8), "f32", (1, 1, 20, 32 @ scope_1.d1, 8), "rmem"
                ]
            )
            T.cast(value_29, value_30, dtype="f32")
            value_31 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 20, 32, 8),
                    "f32",
                    (8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8),
                    "rmem",
                ]
            )
            T.binary(value_26, value_30, value_31, kind=BinaryKind.MUL)
            value_32 = T.tensor_view(
                T.ptr_of(value_22),
                layout=((8 @ scope_1.d0, 4, 20, 32 @ scope_1.d1, 8), (0, 160, 8, 0, 1)),
                shape=(8, 4, 20, 32, 8),
            )
            T.copy(value_31, value_32)
            value_33 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 1, 1, 1),
                    "f32",
                    ((8 @ scope_1.d0, 4, 1, 1, 1, 1, 1), (4, 1, 0, 0, 0, 0, 0)),
                    "rmem",
                ]
            )
            T.reduce(value_32, value_33, axes=(2, 3, 4), keepdim=True, kind=ReduceKind.SUM)
            value_34 = T.tensor_view(
                T.ptr_of(value_21), layout=(8 @ scope_1.d0, 4, 1, 1, 1), shape=(8, 4, 1, 1, 1)
            )
            T.copy(value_33, value_34)
            logit = T.alloc_tensor(
                tensor_type=Tensor[
                    (8, 4, 1, 1, 1), "bf16", (8 @ scope_1.d0, 4, 1, 1, 1), "rmem"
                ]
            )
            T.cast(value_34, logit, dtype="bf16")
            value_35 = T.tensor_view(
                T.ptr_of(out), layout=(62080, 4, 1, 1, 1), shape=(62080, 4, 1, 1, 1)
            )
            window = T.tensor_view(
                T.ptr_of(value_35[(cta.d0 * 32) // 4:(cta.d0 * 32) // 4 + 8, 0:0 + 4, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(8, 4, 1, 1, 1),
                shape=(8, 4, 1, 1, 1),
            )
            T.copy(logit, window)
