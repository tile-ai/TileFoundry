# analysis target=nvidia.h200_sxm module=SCHED_EMBED_T512 function=embed topology=cta wave=132/512
# selection requested=memory executed=memory
# memory traffic=gmem:r20.01KB/w5.02MB@logical,r10.00MB/w2.51GB@total,r20.01KB/w5.02MB@cta,r20.01KB/w5.02MB@thread;rmem:r24B/w0@logical,r12.00KB/w0@total,r24B/w0@cta,r24B/w0@thread;smem:r10.00KB/w10.00KB@logical,r5.00MB/w5.00MB@total,r10.00KB/w10.00KB@cta,r10.00KB/w10.00KB@thread footprint=ids:8B;table:10.00KB;v7:23:10.00KB footprint-precision=lower_bound peak=gmem:2.38GB;rmem:8B;smem:10.00KB persistent=gmem:2.37GB
#   buffer=table holds=20.01KB time=none space=cta.row reuse=1.28MB fits=yes precision=lower_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def embed(
    ids: Tensor[(512,), "i64"], table: Tensor[(248320, 5120), "bf16"], scratch: Tensor[(5120,), "bf16"], out: Tensor[(512, 5120), "bf16"]
):
    with Mesh((Topology("cta", 512),), layout=(512,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        value_1 = T.tensor_view(
            T.ptr_of(scratch[0:0 + 5120]), layout=(1, 5120), shape=(1, 5120)
        )
        value_2 = T.tensor_view(
            0, dtype="bf16", storage="smem", layout=(1, 5120), shape=(1, 5120)
        )
        with Mesh((Topology("thread", 128),), layout=(128,), names=("d0",)) as scope:
            source = T.tensor_view(T.ptr_of(table), layout=(248320, 5120), shape=(248320, 5120))
            tile = T.tensor_view(T.ptr_of(ids[cta.d0:cta.d0 + 1]), layout=(1,), shape=(1,))
            with scope[:] as threads:
                T.copy_async(source, value_2, tile, fill=0.0)
                T.copy(value_2, value_1)
            window = T.tensor_view(
                T.ptr_of(out[cta.d0:cta.d0 + 1, 0:0 + 5120]), layout=(1, 5120), shape=(1, 5120)
            )
            T.copy(value_1, window)
