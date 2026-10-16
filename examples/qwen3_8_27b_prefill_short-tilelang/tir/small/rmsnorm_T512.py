# analysis target=nvidia.h200_sxm module=SCHED_RMSNORM_T512 function=rmsnorm topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r90.00KB/w5.08MB@logical,r11.25MB/w650.00MB@total,r90.00KB/w5.08MB@cta,r40.62KB/w5.04MB@thread;rmem:r610.10KB/w450.05KB@logical,r76.26MB/w56.26MB@total,r610.10KB/w450.05KB@cta,r6.00KB/w4.71KB@thread footprint=gamma:10.00KB;v19:33:40.00KB;v22:34:5.00MB;x:5.00MB footprint-precision=exact peak=gmem:10.05MB;rmem:160.02KB persistent=gmem:5.01MB
#   buffer=gamma holds=10.05MB time=none space=cta.blk reuse=1.24MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def rmsnorm(
    x: Tensor[(512, 5120), "bf16"], gamma: Tensor[(5120,), "bf16"], scratch: Tensor[(20480,), "bf16"], out: Tensor[(512, 5120), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(128,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "bf16",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            value_2 = T.tensor_view(
                T.ptr_of(x[cta.d0 * 4:cta.d0 * 4 + 1, 0:0 + 1]),
                layout=(4, 5120),
                shape=(4, 5120),
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1),
                layout=((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                shape=(4, 5120),
            )
            T.copy(value_2, value_3)
            xf = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "f32",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_3, xf, dtype="f32")
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "f32",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(xf, xf, value_4, kind=BinaryKind.MUL)
            ms = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 1), "f32", ((4 @ scope.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                ]
            )
            T.reduce(value_4, ms, axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 1), "f32", ((4 @ scope.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                ]
            )
            scalar = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar, 1e-06)
            T.binary(ms, scalar, value_5, kind=BinaryKind.ADD)
            value_6 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 1), "f32", ((4 @ scope.d0, 1, 1, 1), (1, 0, 0, 0)), "rmem"
                ]
            )
            T.unary(value_5, value_6, kind=UnaryKind.RSQRT)
            value_7 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "f32",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(xf, value_6, value_7, kind=BinaryKind.MUL)
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5120), "bf16", ((1, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)), "rmem"
                ]
            )
            value_9 = T.tensor_view(
                T.ptr_of(gamma), layout=((1, 5120), (0, 1)), shape=(1, 5120)
            )
            value_10 = T.tensor_view(
                T.ptr_of(value_8),
                layout=((1, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                shape=(1, 5120),
            )
            T.copy(value_9, value_10)
            value_11 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5120), "f32", ((1, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)), "rmem"
                ]
            )
            T.cast(value_10, value_11, dtype="f32")
            value_12 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5120), "f32", ((1, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)), "rmem"
                ]
            )
            scalar_1 = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])
            T.fill(scalar_1, 1.0)
            T.binary(value_11, scalar_1, value_12, kind=BinaryKind.ADD)
            value_13 = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "f32",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(value_7, value_12, value_13, kind=BinaryKind.MUL)
            y = T.alloc_tensor(
                tensor_type=Tensor[
                    (4, 5120),
                    "bf16",
                    ((4 @ scope.d0, 40, 32 @ scope.d1, 4), (0, 4, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_13, y, dtype="bf16")
            value_14 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 20480]), layout=(4, 5120), shape=(4, 5120)
            )
            value_15 = T.tensor_view(T.ptr_of(value_14), layout=(4, 5120), shape=(4, 5120))
            T.copy(y, value_15)
            value_16 = T.tensor_view(T.ptr_of(out), layout=(512, 5120), shape=(512, 5120))
            window = T.tensor_view(
                T.ptr_of(value_16[cta.d0 * 4:cta.d0 * 4 + 4, 0:0 + 5120]),
                layout=(4, 5120),
                shape=(4, 5120),
            )
            T.copy(value_15, window)
