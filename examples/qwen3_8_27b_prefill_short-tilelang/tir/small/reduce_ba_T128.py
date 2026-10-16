# analysis target=nvidia.h200_sxm module=SCHED_REDUCE_BA_T128 function=reduce_ba topology=cta wave=128/128
# selection requested=memory executed=memory
# memory traffic=gmem:r3.19KB/w24.38KB@logical,r408.00KB/w3.05MB@total,r3.19KB/w24.38KB@cta,r224B/w24.19KB@thread;rmem:r3.60KB/w3.56KB@logical,r461.00KB/w456.00KB@total,r3.60KB/w3.56KB@cta,r78B/w38B@thread footprint=p:384.00KB;v10:31:24.00KB;v8:30:192B footprint-precision=exact peak=gmem:408.19KB;rmem:3.38KB persistent=gmem:384.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def reduce_ba(
    p: Tensor[(8, 128, 96), "f32"], scratch: Tensor[(96,), "bf16"], out: Tensor[(128, 96), "bf16"]
):
    with Mesh((Topology("cta", 128),), layout=(128,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 96),), layout=(96,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[(8, 96), "f32", ((8, 96 @ scope.d0), (1, 0)), "rmem"]
            )
            value_2 = T.tensor_view(
                T.ptr_of(p[0:0 + 1, cta.d0:cta.d0 + 1, 0:0 + 1]),
                layout=((8, 96), (12288, 1)),
                shape=(8, 96),
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1), layout=((8, 96 @ scope.d0), (1, 0)), shape=(8, 96)
            )
            T.copy(value_2, value_3)
            total = T.alloc_tensor(
                tensor_type=Tensor[(1, 96), "f32", ((1, 1, 96 @ scope.d0), (0, 0, 1)), "rmem"]
            )
            T.reduce(value_3, total, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
            c = T.alloc_tensor(
                tensor_type=Tensor[(1, 96), "bf16", ((1, 1, 96 @ scope.d0), (0, 0, 1)), "rmem"]
            )
            T.cast(total, c, dtype="bf16")
            value_4 = T.tensor_view(T.ptr_of(scratch[0:0 + 96]), layout=(1, 96), shape=(1, 96))
            value_5 = T.tensor_view(T.ptr_of(value_4), layout=(1, 96), shape=(1, 96))
            T.copy(c, value_5)
            value_6 = T.tensor_view(T.ptr_of(out), layout=(128, 96), shape=(128, 96))
            window = T.tensor_view(
                T.ptr_of(value_6[cta.d0:cta.d0 + 1, 0:0 + 96]), layout=(1, 96), shape=(1, 96)
            )
            T.copy(value_5, window)
