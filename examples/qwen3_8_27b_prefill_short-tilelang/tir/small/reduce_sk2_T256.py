# analysis target=nvidia.h200_sxm module=SCHED_REDUCE_SK_T256 function=reduce_sk topology=cta wave=132/160
# selection requested=memory executed=memory
# memory traffic=gmem:r96.00KB/w2.53MB@logical,r15.00MB/w405.00MB@total,r96.00KB/w2.53MB@cta,r16.08KB/w2.52MB@thread;rmem:r240.05KB/w240.01KB@logical,r37.51MB/w37.50MB@total,r240.05KB/w240.01KB@cta,r288B/w248B@thread footprint=p:64.00KB;r:16.00KB;v17:28:16.00KB footprint-precision=lower_bound peak=gmem:17.52MB;rmem:128.01KB persistent=gmem:12.50MB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def reduce_sk(
    p: Tensor[(2, 1310720), "f32"], r: Tensor[(1310720,), "bf16"], scratch: Tensor[(8192,), "bf16"], out: Tensor[(1310720,), "bf16"]
):
    with Mesh((Topology("cta", 160),), layout=(160,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(T.ptr_of(scratch[0:0 + 8192]), layout=(8192,), shape=(8192,))
        c = T.alloc_tensor(tensor_type=Tensor[(8192,), "bf16", (1024, 8), "rmem"])
        value_2 = T.alloc_tensor(tensor_type=Tensor[(8192,), "f32", (1024, 8), "rmem"])
        value_3 = T.alloc_tensor(
            tensor_type=Tensor[(8192,), "f32", ((1024, 8), (0, 1)), "rmem"]
        )
        value_4 = T.alloc_tensor(
            tensor_type=Tensor[(8192,), "bf16", ((1024, 8), (0, 1)), "rmem"]
        )
        value_5 = T.alloc_tensor(
            tensor_type=Tensor[(8192,), "f32", ((1, 1024, 8), (0, 8, 1)), "rmem"]
        )
        value_6 = T.alloc_tensor(
            tensor_type=Tensor[(8192,), "bf16", ((1, 1024, 8), (0, 8, 1)), "rmem"]
        )
        total = T.alloc_tensor(
            tensor_type=Tensor[(8192,), "f32", ((1, 1024, 8), (0, 8, 1)), "rmem"]
        )
        value_7 = T.alloc_tensor(
            tensor_type=Tensor[(2, 8192), "f32", ((2, 1024, 8), (8, 0, 1)), "rmem"]
        )
        with Mesh((Topology("thread", 1024),), layout=(1024,), names=("d0",)) as scope:
            tile = T.tensor_view(
                T.ptr_of(r[cta.d0 * 8192:cta.d0 * 8192 + 8192]), layout=(8192,), shape=(8192,)
            )
            with scope as threads:
                dst_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 8192]),
                    layout=((1024 @ threads.d0, 8), (0, 1)),
                    shape=(8192,),
                )
                T.copy(tile, dst_frame)
                src_frame = T.tensor_view(
                    T.ptr_of(value_4[0:0 + 8192]),
                    layout=((1024 @ threads.d0, 8), (0, 1)),
                    shape=(8192,),
                )
                dst_frame_1 = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 8192]),
                    layout=((1024 @ threads.d0, 8), (0, 1)),
                    shape=(8192,),
                )
                T.cast(src_frame, dst_frame_1, dtype="f32")
            tile_1 = T.tensor_view(
                T.ptr_of(p[0:0 + 2, cta.d0 * 8192:cta.d0 * 8192 + 8192]),
                layout=((2, 8192), (1310720, 1)),
                shape=(2, 8192),
            )
            with scope as threads_2:
                dst_frame_2 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 2, 0:0 + 8192]),
                    layout=((2, 1024 @ threads_2.d0, 8), (8, 0, 1)),
                    shape=(2, 8192),
                )
                T.copy(tile_1, dst_frame_2)
                src_frame_1 = T.tensor_view(
                    T.ptr_of(value_7[0:0 + 2, 0:0 + 8192]),
                    layout=((2, 1024 @ threads_2.d0, 8), (8, 0, 1)),
                    shape=(2, 8192),
                )
                dst_frame_3 = T.tensor_view(
                    T.ptr_of(total[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                T.reduce(
                    src_frame_1,
                    dst_frame_3,
                    axes=(0,),
                    keepdim=False,
                    kind=ReduceKind.SUM,
                )
                src_frame_2 = T.tensor_view(
                    T.ptr_of(total[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                dst_frame_4 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                T.cast(src_frame_2, dst_frame_4, dtype="bf16")
                src_frame_3 = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                dst_frame_5 = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                T.cast(src_frame_3, dst_frame_5, dtype="f32")
                lhs_frame = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 8192]),
                    layout=((1024 @ threads_2.d0, 8), (0, 1)),
                    shape=(8192,),
                )
                rhs_frame = T.tensor_view(
                    T.ptr_of(value_5[0:0 + 8192]),
                    layout=((1, 1024 @ threads_2.d0, 8), (0, 8, 1)),
                    shape=(8192,),
                )
                dst_frame_6 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8192]),
                    layout=((1024 @ threads_2.d0, 8), (8, 1)),
                    shape=(8192,),
                )
                T.binary(lhs_frame, rhs_frame, dst_frame_6, kind=BinaryKind.ADD)
                src_frame_4 = T.tensor_view(
                    T.ptr_of(value_2[0:0 + 8192]),
                    layout=((1024 @ threads_2.d0, 8), (8, 1)),
                    shape=(8192,),
                )
                dst_frame_7 = T.tensor_view(
                    T.ptr_of(c[0:0 + 8192]),
                    layout=((1024 @ threads_2.d0, 8), (8, 1)),
                    shape=(8192,),
                )
                T.cast(src_frame_4, dst_frame_7, dtype="bf16")
                src_frame_5 = T.tensor_view(
                    T.ptr_of(c[0:0 + 8192]),
                    layout=((1024 @ threads_2.d0, 8), (8, 1)),
                    shape=(8192,),
                )
                T.copy(src_frame_5, value_1)
            window = T.tensor_view(
                T.ptr_of(out[cta.d0 * 8192:cta.d0 * 8192 + 8192]), layout=(8192,), shape=(8192,)
            )
            T.copy(value_1, window)
