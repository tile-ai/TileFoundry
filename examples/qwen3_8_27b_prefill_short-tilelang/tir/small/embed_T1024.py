# analysis target=nvidia.h200_sxm module=SCHED_EMBED_T1024 function=embed topology=cta wave=132/1024
# selection requested=memory executed=memory
# memory traffic=gmem:r20.01KB/w10.02MB@logical,r20.01MB/w10.02GB@total,r20.01KB/w10.02MB@cta,r20.01KB/w10.02MB@thread;rmem:r24B/w0@logical,r24.00KB/w0@total,r24B/w0@cta,r24B/w0@thread;smem:r10.00KB/w10.00KB@logical,r10.00MB/w10.00MB@total,r10.00KB/w10.00KB@cta,r10.00KB/w10.00KB@thread footprint=ids:1.03KB;table:10.00KB;v5:22:10.00KB;v7:23:1.29MB footprint-precision=exact peak=gmem:2.38GB;rmem:8B;smem:10.00KB persistent=gmem:2.37GB
#   buffer=table holds=1.31MB time=none space=cta.row reuse=1.28MB fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def embed(
    ids: Tensor[(1024,), "i64"], table: Tensor[(248320, 5120), "bf16"], scratch: Tensor[(5120,), "bf16"], out: Tensor[(1024, 5120), "bf16"]
):
    with Mesh((Topology("cta", 1024),), layout=(1024,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 128),), layout=(128,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value_2 = T.tensor_view(
                T.ptr_of(table), layout=(248320, 5120), shape=(248320, 5120)
            )
            source = T.tensor_view(
                T.ptr_of(value_2), layout=(248320, 5120), shape=(248320, 5120)
            )
            value_1 = T.tensor_view(
                0, dtype="bf16", storage="smem", layout=(1, 5120), shape=(1, 5120)
            )
            value_3 = T.tensor_view(T.ptr_of(value_1), layout=(1, 5120), shape=(1, 5120))
            value_4 = T.tensor_view(
                T.ptr_of(ids[cta.d0:cta.d0 + 1]), layout=((1,), (0,)), shape=(1,)
            )
            T.copy_async(source, value_3, value_4, fill=0.0)
            value_5 = T.tensor_view(
                T.ptr_of(scratch[0:0 + 5120]), layout=(1, 5120), shape=(1, 5120)
            )
            value_6 = T.tensor_view(T.ptr_of(value_5), layout=(1, 5120), shape=(1, 5120))
            T.copy(value_3, value_6)
            value_7 = T.tensor_view(T.ptr_of(out), layout=(1024, 5120), shape=(1024, 5120))
            window = T.tensor_view(
                T.ptr_of(value_7[cta.d0:cta.d0 + 1, 0:0 + 5120]),
                layout=(1, 5120),
                shape=(1, 5120),
            )
            T.copy(value_6, window)
