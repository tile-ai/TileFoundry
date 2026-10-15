# analysis target=nvidia.h200_sxm module=SCHED_REDUCE_BA_T512 function=reduce_ba topology=cta wave=132/512
# selection requested=memory executed=memory
# memory traffic=gmem:r3.19KB/w96.38KB@logical,r1.59MB/w48.19MB@total,r3.19KB/w96.38KB@cta,r224B/w96.19KB@thread;rmem:r3.60KB/w3.56KB@logical,r1.80MB/w1.78MB@total,r3.60KB/w3.56KB@cta,r78B/w38B@thread footprint=p:3.00KB;v10:31:192B footprint-precision=lower_bound peak=gmem:1.69MB;rmem:3.38KB persistent=gmem:1.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def reduce_ba(
    p: Tensor[(8, 512, 96), "f32"], scratch: Tensor[(96,), "bf16"], out: Tensor[(512, 96), "bf16"]
):
    with Mesh((Topology("cta", 512),), layout=(512,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 96]), layout=(1, 96), shape=(1, 96))
        c = T.alloc_tensor(tensor_type=Tensor[(1, 96), "bf16", ((1, 1, 96), (0, 0, 1)), "rmem"])
        total = T.alloc_tensor(
            tensor_type=Tensor[(1, 96), "f32", ((1, 1, 96), (0, 0, 1)), "rmem"]
        )
        value_2 = T.alloc_tensor(tensor_type=Tensor[(8, 96), "f32", ((8, 96), (1, 0)), "rmem"])
        with Mesh((Topology("thread", 96),), layout=(96,), names=("d0",)) as scope:
            tile = T.tensor_view(
                T.ptr_of(p[0:0 + 8, cta.d0:cta.d0 + 1, 0:0 + 96]),
                layout=((8, 1, 96), (49152, 96, 1)),
                shape=(8, 1, 96),
            )
            tile_1 = T.tensor_view(
                T.ptr_of(tile[0:0 + 8, 0:0 + 1, 0:0 + 96]),
                layout=((8, 96), (49152, 1)),
                shape=(8, 96),
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8, 0:0 + 96]),
                    layout=((8, 96 @ threads.d0), (1, 0)),
                    shape=(8, 96),
                )
                T.copy(tile_1, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8, 0:0 + 96]),
                    layout=((8, 96 @ threads.d0), (1, 0)),
                    shape=(8, 96),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(total[0:0 + 1, 0:0 + 96]),
                    layout=((1, 1, 96 @ threads.d0), (0, 0, 1)),
                    shape=(1, 96),
                )
                T.reduce(src_frame, dst_frame_1, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
                src_frame_1 = T.tensor_view(
                    T.ptr_of(total[0:0 + 1, 0:0 + 96]),
                    layout=((1, 1, 96 @ threads.d0), (0, 0, 1)),
                    shape=(1, 96),
                )
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(c[0:0 + 1, 0:0 + 96]),
                    layout=((1, 1, 96 @ threads.d0), (0, 0, 1)),
                    shape=(1, 96),
                )
                T.cast(src_frame_1, dst_frame_2, dtype="bf16")
                src_frame_2 = T.tensor_view(
                    T.ptr_of(c[0:0 + 1, 0:0 + 96]),
                    layout=((1, 1, 96 @ threads.d0), (0, 0, 1)),
                    shape=(1, 96),
                )
                T.copy(src_frame_2, value_1)
            window = T.tensor_view(
                T.ptr_of(out[cta.d0:cta.d0 + 1, 0:0 + 96]), layout=(1, 96), shape=(1, 96)
            )
            T.copy(value_1, window)
