# analysis target=nvidia.h200_sxm module=SCHED_LM_HEAD function=lm_head topology=cta wave=132/15520
# selection requested=memory executed=memory
# memory traffic=gmem:r170.03KB/w485.06KB@logical,r2.52GB/w7.18GB@total,r170.03KB/w485.06KB@cta,r992B/w485.04KB@thread;rmem:r830.14KB/w830.10KB@logical,r12.29GB/w12.29GB@total,r830.14KB/w830.10KB@cta,r4.12KB/w4.08KB@thread footprint=x:10.00KB footprint-precision=lower_bound peak=gmem:2.37GB;rmem:480.01KB persistent=gmem:2.37GB
#   buffer=x holds=10.00KB time=none space=cta.blk reuse=1.28MB fits=yes precision=lower_bound
#   error="rmem placement peak 480.01KB exceeds capacity 256.00KB"

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def lm_head(
    x: Tensor[(1, 5120), "bf16"], w: Tensor[(248320, 5120), "bf16"], scratch: Tensor[(16,), "bf16"], out: Tensor[(1, 248320), "bf16"]
):
    with Mesh((Topology("cta", 15520),), layout=(15520,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 16]), layout=(1, 16), shape=(1, 16))
        logit = T.alloc_tensor(
            tensor_type=Tensor[(16,), "bf16", ((8, 2, 1, 1, 1), (2, 1, 0, 0, 0)), "rmem"]
        )
        value_2 = T.alloc_tensor(
            tensor_type=Tensor[(16,), "f32", ((8, 2, 1, 1, 1), (2, 1, 0, 0, 0)), "rmem"]
        )
        prod = T.alloc_tensor(
            tensor_type=Tensor[
                (16, 5120), "f32", ((8, 2, 20, 32, 8), (0, 160, 8, 0, 1)), "rmem"
            ]
        )
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[
                (16, 5120), "f32", ((8, 2, 20, 32, 8), (0, 160, 8, 0, 1)), "rmem"
            ]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[
                (16, 5120), "bf16", ((8, 2, 20, 32, 8), (0, 160, 8, 0, 1)), "rmem"
            ]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "f32", ((1, 20, 32, 8), (0, 8, 0, 1)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(1, 5120), "bf16", ((1, 20, 32, 8), (0, 8, 0, 1)), "rmem"]
        )
        with Mesh((Topology("thread", 256),), layout=(8, 32), names=("d0", "d1")) as scope:
            tile = T.tensor_view(
                T.ptr_of(w[cta.d0 * 16:cta.d0 * 16 + 16, 0:0 + 5120]),
                layout=(16, 5120),
                shape=(16, 5120),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                T.copy(tile, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 20, 32 @ threads.d1, 8), (0, 8, 0, 1)),
                    shape=(1, 5120),
                )
                T.copy(x, dst_frame_2)
                src_frame_1 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 20, 32 @ threads.d1, 8), (0, 8, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 20, 32 @ threads.d1, 8), (0, 8, 0, 1)),
                    shape=(1, 5120),
                )
                T.cast(src_frame_1, dst_frame_3, dtype="f32")
                lhs_frame = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 1, 0:0 + 5120]),
                    layout=((1, 20, 32 @ threads.d1, 8), (0, 8, 0, 1)),
                    shape=(1, 5120),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(prod[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                T.binary(lhs_frame, rhs_frame, dst_frame_4, kind=BinaryKind.MUL)
                src_frame_2 = T.tensor_view(
                    T.ptr_of(prod[0:0 + 16, 0:0 + 5120]),
                    layout=((8 @ threads.d0, 2, 20, 32 @ threads.d1, 8), (0, 160, 8, 0, 1)),
                    shape=(16, 5120),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 16]),
                    layout=((8 @ threads.d0, 2, 1, 1, 1), (2, 1, 0, 0, 0)),
                    shape=(16,),
                )
                T.reduce(
                    src_frame_2,
                    dst_frame_5,
                    axes=(-1,),
                    keepdim=False,
                    kind=ReduceKind.SUM,
                )
                src_frame_3 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 16]),
                    layout=((8 @ threads.d0, 2, 1, 1, 1), (2, 1, 0, 0, 0)),
                    shape=(16,),
                )
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(logit[0:0 + 16]),
                    layout=((8 @ threads.d0, 2, 1, 1, 1), (2, 1, 0, 0, 0)),
                    shape=(16,),
                )
                T.cast(src_frame_3, dst_frame_6, dtype="bf16")
            tile_1 = T.tensor_view(
                T.ptr_of(logit[0:0 + 16]), layout=((1, 8, 2), (0, 2, 1)), shape=(1, 16)
            )
            with scope[:] as threads_7:
                T.copy(tile_1, value_1)
            window = T.tensor_view(
                T.ptr_of(out[0:0 + 1, cta.d0 * 16:cta.d0 * 16 + 16]),
                layout=((1, 16), (248320, 1)),
                shape=(1, 16),
            )
            T.copy(value_1, window)
