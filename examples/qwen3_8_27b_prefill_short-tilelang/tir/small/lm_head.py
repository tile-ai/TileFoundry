# analysis target=nvidia.h200_sxm module=SCHED_LM_HEAD function=lm_head topology=cta wave=132/15520
# selection requested=memory executed=memory
# memory traffic=gmem:r170.03KB/w485.06KB@logical,r2.52GB/w7.18GB@total,r170.03KB/w485.06KB@cta,r992B/w485.04KB@thread;rmem:r830.14KB/w830.10KB@logical,r12.29GB/w12.29GB@total,r830.14KB/w830.10KB@cta,r4.12KB/w4.08KB@thread footprint=v14:28:32B;v16:29:4.12KB;w:20.62MB;x:10.00KB footprint-precision=exact peak=gmem:2.37GB;rmem:480.01KB persistent=gmem:2.37GB
#   buffer=x holds=20.64MB time=none space=cta.blk reuse=1.28MB fits=yes precision=exact
#   error="rmem placement peak 480.01KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def lm_head(
    x: Tensor[(1, 5120), "bf16"], w: Tensor[(248320, 5120), "bf16"], scratch: Tensor[(16,), "bf16"], out: Tensor[(1, 248320), "bf16"]
):
    with Mesh((Topology("cta", 15520),), layout=(15520,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(8, 32), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            value_1 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 5120),
                    "bf16",
                    ((8 @ scope.d0, 2, 20, 32 @ scope.d1, 8), (0, 160, 8, 0, 1)),
                    "rmem",
                ]
            )
            value_2 = T.tensor_view(
                T.ptr_of(w[cta.d0 * 16:cta.d0 * 16 + 1, 0:0 + 1]),
                layout=(16, 5120),
                shape=(16, 5120),
            )
            value_3 = T.tensor_view(
                T.ptr_of(value_1),
                layout=((8 @ scope.d0, 2, 20, 32 @ scope.d1, 8), (0, 160, 8, 0, 1)),
                shape=(16, 5120),
            )
            T.copy(value_2, value_3)
            value_4 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 5120),
                    "f32",
                    ((8 @ scope.d0, 2, 20, 32 @ scope.d1, 8), (0, 160, 8, 0, 1)),
                    "rmem",
                ]
            )
            T.cast(value_3, value_4, dtype="f32")
            value_5 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5120), "bf16", ((1, 20, 32 @ scope.d1, 8), (0, 8, 0, 1)), "rmem"
                ]
            )
            value_6 = T.tensor_view(T.ptr_of(x), layout=(1, 5120), shape=(1, 5120))
            value_7 = T.tensor_view(
                T.ptr_of(value_5),
                layout=((1, 20, 32 @ scope.d1, 8), (0, 8, 0, 1)),
                shape=(1, 5120),
            )
            T.copy(value_6, value_7)
            value_8 = T.alloc_tensor(
                tensor_type=Tensor[
                    (1, 5120), "f32", ((1, 20, 32 @ scope.d1, 8), (0, 8, 0, 1)), "rmem"
                ]
            )
            T.cast(value_7, value_8, dtype="f32")
            prod = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 5120),
                    "f32",
                    ((8 @ scope.d0, 2, 20, 32 @ scope.d1, 8), (0, 160, 8, 0, 1)),
                    "rmem",
                ]
            )
            T.binary(value_4, value_8, prod, kind=BinaryKind.MUL)
            value_9 = T.alloc_tensor(
                tensor_type=Tensor[
                    (16,), "f32", ((8 @ scope.d0, 2, 1, 1, 1), (2, 1, 0, 0, 0)), "rmem"
                ]
            )
            T.reduce(prod, value_9, axes=(-1,), keepdim=False, kind=ReduceKind.SUM)
            logit = T.alloc_tensor(
                tensor_type=Tensor[
                    (16,), "bf16", ((8 @ scope.d0, 2, 1, 1, 1), (2, 1, 0, 0, 0)), "rmem"
                ]
            )
            T.cast(value_9, logit, dtype="bf16")
            value_10 = T.tensor_view(
                T.ptr_of(logit), layout=((1, 8 @ scope.d0, 2), (0, 2, 1)), shape=(1, 16)
            )
            value_11 = T.tensor_view(T.ptr_of(scratch[0:0 + 16]), layout=(1, 16), shape=(1, 16))
            value_12 = T.tensor_view(T.ptr_of(value_11), layout=(1, 16), shape=(1, 16))
            T.copy(value_10, value_12)
            value_13 = T.tensor_view(T.ptr_of(out), layout=(1, 248320), shape=(1, 248320))
            window = T.tensor_view(
                T.ptr_of(value_13[0:0 + 1, cta.d0 * 16:cta.d0 * 16 + 16]),
                layout=((1, 16), (248320, 1)),
                shape=(1, 16),
            )
            T.copy(value_12, window)
