# analysis target=nvidia.h200_sxm module=RMSNorm function=rmsnorm topology=cta wave=132/512
# selection requested=memory executed=memory
# memory traffic=gmem:r30.00KB/w10.02MB@logical,r15.00MB/w5.01GB@total,r30.00KB/w10.02MB@cta,r120B/w10.00MB@thread;rmem:r330.12KB/w290.04KB@logical,r165.06MB/w145.02MB@total,r330.12KB/w290.04KB@cta,r1.41KB/w1.17KB@thread footprint=gamma:10.00KB;hidden:2.58MB;v24:44:2.58MB footprint-precision=exact peak=gmem:20.01MB;rmem:120.01KB persistent=gmem:10.01MB
#   buffer=gamma holds=5.17MB time=none space=cta.rb reuse=1.28MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def rmsnorm(
    hidden: Tensor[(1, 1024, 5120), "bf16"], gamma: ConstTensor[(5120,), "bf16"], out: Tensor[(1024, 5, 256, 4), "bf16"]
):
    with Mesh((Topology("cta", 512),), layout=(512,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(256,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            value_1 = T.tensor_view(
                T.ptr_of(hidden[0:0 + 1, cta.d0 * 2:cta.d0 * 2 + 1, 0:0 + 1]),
                layout=(2, 5, 256, 4),
                shape=(2, 5, 256, 4),
            )
            value_2 = T.tensor_view(
                T.ptr_of(value), layout=(2, 5, 256 @ scope.d0, 4), shape=(2, 5, 256, 4)
            )
            T.copy(value_1, value_2)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_2, x32, dtype="f32")
            value_3 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[
                    (2, 5, 256, 4), "f32", ((2, 5, 256 @ scope.d0, 4), (20, 4, 0, 1)), "rmem"
                ]
            )
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.unary(x32, value_5, kind=UnaryKind.SQUARE)
            value_6 = T.tensor_view(
                T.ptr_of(value_4),
                layout=((2, 5, 256 @ scope.d0, 4), (20, 4, 0, 1)),
                shape=(2, 5, 256, 4),
            )
            T.copy(value_5, value_6)
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (2, 1, 1, 1),
                    "f32",
                    ((2, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)) @ {B()},
                    "rmem",
                ]
            )
            T.reduce(value_6, value_7, axes=(1, 2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_8 = T.tensor_view(T.ptr_of(value_3), layout=(2, 1, 1, 1), shape=(2, 1, 1, 1))
            T.copy(value_7, value_8)
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(value_8, scalar, value_9, kind=BinaryKind.ADD)
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            T.unary(value_9, value_10, kind=UnaryKind.RSQRT)
            value_11 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.binary(x32, value_10, value_11, kind=BinaryKind.MUL)
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            value_13 = T.tensor_view(
                T.ptr_of(gamma), layout=((1, 5, 256, 4), (0, 1024, 4, 1)), shape=(1, 5, 256, 4)
            )
            value_14 = T.tensor_view(
                T.ptr_of(value_12), layout=(1, 5, 256 @ scope.d0, 4), shape=(1, 5, 256, 4)
            )
            T.copy(value_13, value_14)
            value_15 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_14, value_15, dtype="f32")
            value_16 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.d0, 4), "rmem"]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_15, scalar_1, value_16, kind=BinaryKind.ADD)
            value_17 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.binary(value_11, value_16, value_17, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256 @ scope.d0, 4), "rmem"]
            )
            T.cast(value_17, y, dtype="bf16")
            value_18 = T.tensor_view(
                T.ptr_of(out), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            window = T.tensor_view(
                T.ptr_of(value_18[cta.d0 * 2:cta.d0 * 2 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(2, 5, 256, 4),
                shape=(2, 5, 256, 4),
            )
            T.copy(y, window)
