# analysis target=nvidia.h200_sxm module=CapturedInsertUpdate function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r32B/w0@logical,r32B/w0@total,r32B/w0@cta,r32B/w0@thread;rmem:r48B/w0@logical,r48B/w0@total,r48B/w0@cta,r48B/w0@thread;smem:r16B/w48B@logical,r16B/w48B@total,r16B/w48B@cta,r16B/w48B@thread footprint=idx:8B;x:8B footprint-precision=exact peak=gmem:48B;rmem:0;smem:24B persistent=gmem:48B
#   buffer=x holds=16B time=r space=none reuse=8B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(4, 4), "bf16"], idx: Tensor[(2,), "i64"], out: Tensor[(2, 4), "bf16", (2, 4), "smem"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            T.fill(value, 0.0)
            for r in range(0, 2, 1):
                value_1 = T.tensor_view(T.ptr_of(x), layout=(4, 4), shape=(4, 4))
                source = T.tensor_view(T.ptr_of(value_1), layout=(4, 4), shape=(4, 4))
                row_dst = T.tensor_view(
                    16, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                value_2 = T.tensor_view(T.ptr_of(row_dst), layout=(1, 4), shape=(1, 4))
                value_3 = T.tensor_view(T.ptr_of(idx), layout=(2,), shape=(2,))
                value_4 = T.tensor_view(
                    T.ptr_of(value_3[r:r + 1]), layout=((1,), (0,)), shape=(1,)
                )
                T.copy_async(source, value_2, value_4, fill=0)
                with scope[:] as scope_1:
                    row_dst_2 = T.tensor_view(
                        16, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                    )
                    value_5 = T.tensor_view(T.ptr_of(row_dst_2), layout=(1, 4), shape=(1, 4))
                    value_6 = T.tensor_view(T.ptr_of(value_5), layout=(1, 4), shape=(1, 4))
                    value_7 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                    value_8 = T.tensor_view(T.ptr_of(value_7), layout=(2, 4), shape=(2, 4))
                    value_9 = T.tensor_view(T.ptr_of(value_8), layout=(2, 4), shape=(2, 4))
                    window = T.tensor_view(
                        T.ptr_of(value_9[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy(value_6, window)
