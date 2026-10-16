# analysis target=nvidia.h200_sxm module=DeltaRuleIntra function=intra_g topology=cta wave=132/768
# selection requested=memory executed=memory
# memory traffic=gmem:r768B/w192.25KB@logical,r576.00KB/w144.19MB@total,r768B/w192.25KB@cta,r264B/w192.00KB@thread;rmem:r50.05KB/w50.01KB@logical,r37.54MB/w37.51MB@total,r50.05KB/w50.01KB@cta,r1.33KB/w1.28KB@thread;smem:r256B/w256B@logical,r192.00KB/w192.00KB@total,r256B/w256B@cta,r256B/w4B@thread footprint=g:33.00KB;iota:256B;v28:70:33.00KB footprint-precision=exact peak=gmem:384.25KB;rmem:32.02KB;smem:256B persistent=gmem:192.25KB
#   buffer=iota holds=66.25KB time=none space=cta.c,cta.h reuse=32.75KB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def intra_g(
    g: Tensor[(1024, 48), "f32"], iota: Tensor[(64,), "i32"], out: Tensor[(48, 1024, 1), "f32"]
):
    with Mesh((Topology("cta", 768),), layout=(16, 48), names=("d0", "d1")) as cta:
        with Mesh((Topology("thread", 128),), layout=(2, 64), names=("d0", "d1")) as scope:
            T.fill(out, 0.0)
            with Mesh((Topology("thread", 128),), layout=(64,), names=("d0",)) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "f32", (64 @ scope_1.d0, 1), "rmem"]
                )
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 64), "f32", ((64 @ scope_1.d0, 64), (0, 1)), "rmem"]
                )
                value_2 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", (1, 64), "rmem"])
                value_4 = T.alloc_tensor(
                    tensor_type=Tensor[(64,), "f32", (64 @ scope_1.d0,), "rmem"]
                )
                value_5 = T.tensor_view(
                    T.ptr_of(g[cta.d0 * 64:cta.d0 * 64 + 1, cta.d1:cta.d1 + 1]),
                    layout=((64,), (48,)),
                    shape=(64,),
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_4), layout=(64 @ scope_1.d0,), shape=(64,)
                )
                T.copy(value_5, value_6)
                value_3 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                value_7 = T.tensor_view(T.ptr_of(value_3), layout=(64,), shape=(64,))
                T.copy(value_6, value_7)
                value_8 = T.tensor_view(
                    T.ptr_of(value_3), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_9 = T.tensor_view(T.ptr_of(value_2), layout=(1, 64), shape=(1, 64))
                T.copy(value_8, value_9)
                value_10 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "i32", (1, 64), "rmem"])
                value_11 = T.tensor_view(
                    T.ptr_of(iota), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_12 = T.tensor_view(T.ptr_of(value_10), layout=(1, 64), shape=(1, 64))
                T.copy(value_11, value_12)
                value_13 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "i32", (64 @ scope_1.d0, 1), "rmem"]
                )
                value_14 = T.tensor_view(
                    T.ptr_of(iota), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_15 = T.tensor_view(
                    T.ptr_of(value_13), layout=(64 @ scope_1.d0, 1), shape=(64, 1)
                )
                T.copy(value_14, value_15)
                value_16 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "bool", ((64 @ scope_1.d0, 64), (64, 1)), "rmem"
                    ]
                )
                T.binary(value_12, value_15, value_16, kind=BinaryKind.LE)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "f32", ((64 @ scope_1.d0, 64), (64, 1)), "rmem"
                    ]
                )
                T.cast(value_16, value_17, dtype="f32")
                value_18 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "f32", ((64 @ scope_1.d0, 64), (64, 1)), "rmem"
                    ]
                )
                T.binary(value_9, value_17, value_18, kind=BinaryKind.MUL)
                value_19 = T.tensor_view(
                    T.ptr_of(value_1), layout=((64 @ scope_1.d0, 64), (0, 1)), shape=(64, 64)
                )
                T.copy(value_18, value_19)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "f32", ((64 @ scope_1.d0, 1), (1, 0)), "rmem"]
                )
                T.reduce(value_19, value_20, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                value_21 = T.tensor_view(
                    T.ptr_of(value), layout=(64 @ scope_1.d0, 1), shape=(64, 1)
                )
                T.copy(value_20, value_21)
                value_22 = T.tensor_view(
                    T.ptr_of(value), layout=(1, 64 @ scope_1.d0, 1), shape=(1, 64, 1)
                )
                value_23 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 1024, 1), shape=(48, 1024, 1)
                )
                window = T.tensor_view(
                    T.ptr_of(value_23[cta.d1:cta.d1 + 1, cta.d0 * 64:cta.d0 * 64 + 64, 0:0 + 1]),
                    layout=((1, 64, 1), (1024, 1, 1)),
                    shape=(1, 64, 1),
                )
                T.copy(value_22, window)
