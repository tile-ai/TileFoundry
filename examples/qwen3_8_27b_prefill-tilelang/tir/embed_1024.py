# analysis target=nvidia.h200_sxm module=Embed function=embed topology=cta wave=132/256
# selection requested=memory executed=memory
# memory traffic=gmem:r40.03KB/w10.04MB@logical,r10.01MB/w2.51GB@total,r40.03KB/w10.04MB@cta,r192B/w10.00MB@thread;rmem:r40.33KB/w40.04KB@logical,r10.09MB/w10.02MB@total,r40.38KB/w40.06KB@cta,r544B/w224B@thread footprint=<value 0>[0]:8B;table:10.00KB;v10:45:1.29MB footprint-precision=upper_bound peak=gmem:2.38GB;rmem:10.01KB persistent=gmem:2.37GB
#   buffer=table holds=1.30MB time=none space=cta.b reuse=1.28MB fits=yes precision=upper_bound
#   buffer=<value 0>[0] holds=1.30MB time=none space=cta.b reuse=1.02KB fits=yes precision=upper_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def embed(
    table: ConstTensor[(248320, 5120), "bf16"], token_ids: Tensor[(1024,), "i64"], out: Tensor[(1024, 5, 256, 4), "bf16"]
):
    with Mesh((Topology("cta", 256),), layout=(256,), names=("b",)) as cta:
        with Mesh((Topology("thread", 256),), layout=(256,), names=("t",)) as scope:
            value = T.tensor_view(
                T.ptr_of(out), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
            )
            T.fill(value, 0.0)
            for i in range(0, 4, 1):
                value_1 = T.alloc_tensor(
                    tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256 @ scope.t, 4), "rmem"]
                )
                value_2 = T.tensor_view(T.ptr_of(token_ids), layout=(1024,), shape=(1024,))
                value_3 = T.tensor_view(
                    T.ptr_of(value_2[(cta.b * 4) + i:(cta.b * 4) + i + 1]),
                    layout=((1,), (0,)),
                    shape=(1,),
                )
                value_4 = T.tensor_view(T.ptr_of(value_3), layout=(), shape=())
                value_5 = T.tensor_view(
                    T.ptr_of(table), layout=(248320, 5120), shape=(248320, 5120)
                )
                value_6 = T.tensor_view(
                    T.ptr_of(value_5), layout=(248320, 5, 256, 4), shape=(248320, 5, 256, 4)
                )
                value_7 = T.tensor_view(
                    T.ptr_of(value_6), layout=(248320, 5, 256, 4), shape=(248320, 5, 256, 4)
                )
                value_8 = T.tensor_view(
                    T.ptr_of(value_7[value_4:value_4 + 1, 0:0 + 1, 0:0 + 1, 0:0 + 1]),
                    layout=((1, 5, 256, 4), (0, 1024, 4, 1)),
                    shape=(1, 5, 256, 4),
                )
                value_9 = T.tensor_view(
                    T.ptr_of(value_1), layout=(1, 5, 256 @ scope.t, 4), shape=(1, 5, 256, 4)
                )
                dst_view = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ scope.t, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(value_8, dst_view)
                value_view = T.tensor_view(
                    T.ptr_of(value_9[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256 @ scope.t, 4),
                    shape=(1, 5, 256, 4),
                )
                value_10 = T.tensor_view(
                    T.ptr_of(out), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
                )
                value_11 = T.tensor_view(
                    T.ptr_of(value_10), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
                )
                value_12 = T.tensor_view(
                    T.ptr_of(value_11), layout=(1024, 5, 256, 4), shape=(1024, 5, 256, 4)
                )
                window = T.tensor_view(
                    T.ptr_of(value_12[(cta.b * 4) + i:(cta.b * 4) + i + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256, 4),
                    shape=(1, 5, 256, 4),
                )
                T.copy(value_view, window)
