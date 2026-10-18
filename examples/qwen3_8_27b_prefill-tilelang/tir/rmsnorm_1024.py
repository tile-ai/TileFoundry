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
    with Mesh((Topology("cta", 512),), layout=(512,), names=("rb",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(256,), names=("t",)) as scope:
            value = T.tensor_view(
                T.ptr_of(out), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            T.fill(value, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            value_2 = T.tensor_view(
                T.ptr_of(hidden), layout=(1, 1024, 5120), shape=(1, 1024, 5120)
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_2), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            value_4 = T.tensor_view(
                T.ptr_of(value_3[cta.rb * 2:cta.rb * 2 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                layout=(2, 5, 256, 4),
                shape=(2, 5, 256, 4),
            )
            value_5 = T.tensor_view(
                T.ptr_of(value_1), layout=(2, 5, 256 @ scope.t, 4), shape=(2, 5, 256, 4)
            )
            dst_view = T.tensor_view(
                T.ptr_of(value_5[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(2, 5, 256 @ scope.t, 4),
                shape=(2, 5, 256, 4),
            )
            T.copy(value_4, dst_view)
            x32 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            src_view = T.tensor_view(
                T.ptr_of(value_5[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(2, 5, 256 @ scope.t, 4),
                shape=(2, 5, 256, 4),
            )
            T.cast(src_view, x32, dtype="f32")
            value_6 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (2, 5, 256, 4), "f32", ((2, 5, 256 @ scope.t, 4), (20, 4, 0, 1)), "rmem"
                ]
            )
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.unary(x32, value_8, kind=UnaryKind.SQUARE)
            value_9 = T.tensor_view(
                T.ptr_of(value_7),
                layout=((2, 5, 256 @ scope.t, 4), (20, 4, 0, 1)),
                shape=(2, 5, 256, 4),
            )
            dst_view_1 = T.tensor_view(
                T.ptr_of(value_9[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=((2, 5, 256 @ scope.t, 4), (20, 4, 0, 1)),
                shape=(2, 5, 256, 4),
            )
            T.copy(value_8, dst_view_1)
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[
                    (2, 1, 1, 1),
                    "f32",
                    ((2, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)) @ {B()},
                    "rmem",
                ]
            )
            src_view_1 = T.tensor_view(
                T.ptr_of(value_9[0:0 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=((2, 5, 256 @ scope.t, 4), (20, 4, 0, 1)),
                shape=(2, 5, 256, 4),
            )
            T.reduce(src_view_1, value_10, axes=(1, 2, 3), keepdim=True, kind=ReduceKind.MEAN)
            value_11 = T.tensor_view(T.ptr_of(value_6), layout=(2, 1, 1, 1), shape=(2, 1, 1, 1))
            T.copy(value_10, value_11)
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(value_11, scalar, value_12, kind=BinaryKind.ADD)
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[(2, 1, 1, 1), "f32", (2, 1, 1, 1), "rmem"]
            )
            T.unary(value_12, value_13, kind=UnaryKind.RSQRT)
            value_14 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.binary(x32, value_13, value_14, kind=BinaryKind.MUL)
            value_15 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            value_16 = T.tensor_view(T.ptr_of(gamma), layout=(5120,), shape=(5120,))
            value_17 = T.tensor_view(
                T.ptr_of(value_16),
                layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                shape=(1, 5, 256, 4),
            )
            value_18 = T.tensor_view(
                T.ptr_of(value_15), layout=(1, 5, 256 @ scope.t, 4), shape=(1, 5, 256, 4)
            )
            dst_view_2 = T.tensor_view(
                T.ptr_of(value_18[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.copy(value_17, dst_view_2)
            value_19 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            src_view_2 = T.tensor_view(
                T.ptr_of(value_18[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(1, 5, 256 @ scope.t, 4),
                shape=(1, 5, 256, 4),
            )
            T.cast(src_view_2, value_19, dtype="f32")
            value_20 = T.alloc_tensor(
                tensor_type=Tensor[(1, 5, 256, 4), "f32", (1, 5, 256 @ scope.t, 4), "rmem"]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_19, scalar_1, value_20, kind=BinaryKind.ADD)
            value_21 = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "f32", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.binary(value_14, value_20, value_21, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[(2, 5, 256, 4), "bf16", (2, 5, 256 @ scope.t, 4), "rmem"]
            )
            T.cast(value_21, y, dtype="bf16")
            value_22 = T.tensor_view(
                T.ptr_of(out), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            window = T.tensor_view(
                T.ptr_of(value_22[cta.rb * 2:cta.rb * 2 + 2, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                layout=(2, 5, 256, 4),
                shape=(2, 5, 256, 4),
            )
            T.copy(y, window)
