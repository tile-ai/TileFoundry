# analysis target=nvidia.h200_sxm module=SCHED_REDUCE_SK_T256 function=reduce_sk topology=cta wave=132/160
# selection requested=memory executed=memory
# memory traffic=gmem:r96.00KB/w2.53MB@logical,r15.00MB/w405.00MB@total,r96.00KB/w2.53MB@cta,r16.08KB/w2.52MB@thread;rmem:r240.05KB/w240.01KB@logical,r37.51MB/w37.50MB@total,r240.05KB/w240.01KB@cta,r288B/w248B@thread footprint=p:8.25MB;r:2.06MB;v15:27:16.00KB;v17:28:2.06MB footprint-precision=exact peak=gmem:15.02MB;rmem:128.01KB persistent=gmem:12.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def reduce_sk(
    p: Tensor[(2, 1310720), "f32"], r: Tensor[(1310720,), "bf16"], scratch: Tensor[(8192,), "bf16"], out: Tensor[(1310720,), "bf16"]
):
    with Mesh((Topology("cta", 160),), layout=(160,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1024),), layout=(1024,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "bf16", ((1024 @ scope.d0, 8), (0, 1)), "rmem"]
            )
            value_2 = T.tensor_view(
                T.ptr_of(r[cta.d0 * 8192:cta.d0 * 8192 + 1]), layout=(8192,), shape=(8192,)
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1), layout=((1024 @ scope.d0, 8), (0, 1)), shape=(8192,)
            )
            T.copy(value_2, value_3)
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "f32", ((1024 @ scope.d0, 8), (0, 1)), "rmem"]
            )
            T.cast(value_3, value_4, dtype="f32")
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (2, 8192), "f32", ((2, 1024 @ scope.d0, 8), (8, 0, 1)), "rmem"
                ]
            )
            value_6 = T.tensor_view(
                T.ptr_of(p[0:0 + 1, cta.d0 * 8192:cta.d0 * 8192 + 1]),
                layout=((2, 8192), (1310720, 1)),
                shape=(2, 8192),
            )
            value_7 = T.tensor_view(
                T.ptr_of(value_5), layout=((2, 1024 @ scope.d0, 8), (8, 0, 1)), shape=(2, 8192)
            )
            T.copy(value_6, value_7)
            total = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "f32", ((1, 1024 @ scope.d0, 8), (0, 8, 1)), "rmem"]
            )
            T.reduce(value_7, total, axes=(0,), keepdim=False, kind=ReduceKind.SUM)
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[
                    (8192,), "bf16", ((1, 1024 @ scope.d0, 8), (0, 8, 1)), "rmem"
                ]
            )
            T.cast(total, value_8, dtype="bf16")
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "f32", ((1, 1024 @ scope.d0, 8), (0, 8, 1)), "rmem"]
            )
            T.cast(value_8, value_9, dtype="f32")
            value_10 = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "f32", ((1024 @ scope.d0, 8), (8, 1)), "rmem"]
            )
            T.binary(value_4, value_9, value_10, kind=BinaryKind.ADD)
            c = T.alloc_tensor(
                tensor_type=Tensor[(8192,), "bf16", ((1024 @ scope.d0, 8), (8, 1)), "rmem"]
            )
            T.cast(value_10, c, dtype="bf16")
            value_11 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 8192]), layout=(8192,), shape=(8192,)
            )
            value_12 = T.tensor_view(T.ptr_of(value_11), layout=(8192,), shape=(8192,))
            T.copy(c, value_12)
            value_13 = T.tensor_view(T.ptr_of(out), layout=(1310720,), shape=(1310720,))
            window = T.tensor_view(
                T.ptr_of(value_13[cta.d0 * 8192:cta.d0 * 8192 + 8192]),
                layout=(8192,),
                shape=(8192,),
            )
            T.copy(value_12, window)
