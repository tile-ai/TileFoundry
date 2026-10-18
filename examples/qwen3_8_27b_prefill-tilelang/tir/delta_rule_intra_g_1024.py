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
    with Mesh((Topology("cta", 768),), layout=(16, 48), names=("c", "h")) as cta:
        with Mesh((Topology("thread", 128),), layout=(2, 64), names=("half", "i")) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(48, 1024, 1), shape=(48, 1024, 1))
            T.fill(value, 0.0)
            with Mesh((Topology("thread", 128),), layout=(64,), names=("i",)) as scope_1:
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "f32", (64 @ scope_1.i, 1), "rmem"]
                )
                value_2 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 64), "f32", ((64 @ scope_1.i, 64), (0, 1)), "rmem"]
                )
                value_3 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "f32", (1, 64), "rmem"])
                value_5 = T.alloc_tensor(
                    tensor_type=Tensor[(64,), "f32", (64 @ scope_1.i,), "rmem"]
                )
                value_6 = T.tensor_view(T.ptr_of(g), layout=(1024, 48), shape=(1024, 48))
                value_7 = T.tensor_view(
                    T.ptr_of(value_6[cta.c * 64:cta.c * 64 + 1, cta.h:cta.h + 1]),
                    layout=((64, 1), (48, 0)),
                    shape=(64, 1),
                )
                value_8 = T.tensor_view(T.ptr_of(value_7), layout=((64,), (48,)), shape=(64,))
                value_9 = T.tensor_view(
                    T.ptr_of(value_5), layout=(64 @ scope_1.i,), shape=(64,)
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64]), layout=(64 @ scope_1.i,), shape=(64,)
                )
                T.copy(value_8, dst_view)
                src_view = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 64]), layout=(64 @ scope_1.i,), shape=(64,)
                )
                value_4 = T.tensor_view(
                    0, dtype="f32", storage="smem", layout=(64,), shape=(64,)
                )
                value_10 = T.tensor_view(T.ptr_of(value_4), layout=(64,), shape=(64,))
                T.copy(src_view, value_10)
                value_11 = T.tensor_view(
                    T.ptr_of(value_10), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_12 = T.tensor_view(T.ptr_of(value_3), layout=(1, 64), shape=(1, 64))
                T.copy(value_11, value_12)
                value_13 = T.alloc_tensor(tensor_type=Tensor[(1, 64), "i32", (1, 64), "rmem"])
                value_14 = T.tensor_view(T.ptr_of(iota), layout=(64,), shape=(64,))
                value_15 = T.tensor_view(
                    T.ptr_of(value_14), layout=((1, 64), (0, 1)), shape=(1, 64)
                )
                value_16 = T.tensor_view(T.ptr_of(value_13), layout=(1, 64), shape=(1, 64))
                T.copy(value_15, value_16)
                value_17 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "i32", (64 @ scope_1.i, 1), "rmem"]
                )
                value_18 = T.tensor_view(
                    T.ptr_of(value_14), layout=((64, 1), (1, 0)), shape=(64, 1)
                )
                value_19 = T.tensor_view(
                    T.ptr_of(value_17), layout=(64 @ scope_1.i, 1), shape=(64, 1)
                )
                dst_view_1 = T.tensor_view(
                    T.ptr_of(value_19[0:0 + 64, 0:0 + 1]),
                    layout=(64 @ scope_1.i, 1),
                    shape=(64, 1),
                )
                T.copy(value_18, dst_view_1)
                value_20 = T.alloc_tensor(
                    tensor_type=Tensor[
                        (64, 64), "bool", ((64 @ scope_1.i, 64), (64, 1)), "rmem"
                    ]
                )
                rhs_view = T.tensor_view(
                    T.ptr_of(value_19[0:0 + 64, 0:0 + 1]),
                    layout=(64 @ scope_1.i, 1),
                    shape=(64, 1),
                )
                T.binary(value_16, rhs_view, value_20, kind=BinaryKind.LE)
                value_21 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 64), "f32", ((64 @ scope_1.i, 64), (64, 1)), "rmem"]
                )
                T.cast(value_20, value_21, dtype="f32")
                value_22 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 64), "f32", ((64 @ scope_1.i, 64), (64, 1)), "rmem"]
                )
                T.binary(value_12, value_21, value_22, kind=BinaryKind.MUL)
                value_23 = T.tensor_view(
                    T.ptr_of(value_2), layout=((64 @ scope_1.i, 64), (0, 1)), shape=(64, 64)
                )
                dst_view_2 = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                    layout=((64 @ scope_1.i, 64), (0, 1)),
                    shape=(64, 64),
                )
                T.copy(value_22, dst_view_2)
                value_24 = T.alloc_tensor(
                    tensor_type=Tensor[(64, 1), "f32", ((64 @ scope_1.i, 1), (1, 0)), "rmem"]
                )
                src_view_1 = T.tensor_view(
                    T.ptr_of(value_23[0:0 + 64, 0:0 + 64]),
                    layout=((64 @ scope_1.i, 64), (0, 1)),
                    shape=(64, 64),
                )
                T.reduce(src_view_1, value_24, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
                value_25 = T.tensor_view(
                    T.ptr_of(value_1), layout=(64 @ scope_1.i, 1), shape=(64, 1)
                )
                dst_view_3 = T.tensor_view(
                    T.ptr_of(value_25[0:0 + 64, 0:0 + 1]),
                    layout=(64 @ scope_1.i, 1),
                    shape=(64, 1),
                )
                T.copy(value_24, dst_view_3)
                value_26 = T.tensor_view(
                    T.ptr_of(value_25), layout=(1, 64 @ scope_1.i, 1), shape=(1, 64, 1)
                )
                value_view = T.tensor_view(
                    T.ptr_of(value_26[0:0 + 1, 0:0 + 64, 0:0 + 1]),
                    layout=(1, 64 @ scope_1.i, 1),
                    shape=(1, 64, 1),
                )
                value_27 = T.tensor_view(
                    T.ptr_of(out), layout=(48, 1024, 1), shape=(48, 1024, 1)
                )
                window = T.tensor_view(
                    T.ptr_of(value_27[cta.h:cta.h + 1, cta.c * 64:cta.c * 64 + 64, 0:0 + 1]),
                    layout=((1, 64, 1), (1024, 1, 1)),
                    shape=(1, 64, 1),
                )
                T.copy(value_view, window)
