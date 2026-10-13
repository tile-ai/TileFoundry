# analysis target=nvidia.h200_sxm module=Embed function=embed topology=cta wave=132/256
# selection requested=memory executed=memory
# memory traffic=gmem:r40.03KB/w10.04MB@logical,r10.01MB/w2.51GB@total,r40.03KB/w10.04MB@cta,r192B/w10.00MB@thread;rmem:r40.33KB/w40.04KB@logical,r10.09MB/w10.02MB@total,r40.38KB/w40.06KB@cta,r544B/w224B@thread footprint=<value 0>[0]:32B;table:10.00KB;v9:45:10.00KB footprint-precision=upper_bound peak=gmem:2.38GB;rmem:10.01KB persistent=gmem:2.37GB
#   buffer=table holds=20.03KB time=i space=cta.b reuse=5.15MB fits=yes precision=upper_bound
#   buffer=<value 0>[0] holds=20.03KB time=none space=cta.b reuse=4.09KB fits=yes precision=upper_bound

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def embed(
    table: Tensor[(248320, 5120), "bf16"], token_ids: Tensor[(1024,), "i64"], out: Tensor[(1024, 5, 256, 4), "bf16"]
):
    with Mesh((Topology("cta", 256),), layout=(256,), names=("d0",)) as cta:
        T.fill(out, 0.0)
        row = T.alloc_tensor(tensor_type=Tensor[(1, 5, 256, 4), "bf16", (1, 5, 256, 4), "rmem"])
        with Mesh((Topology("thread", 256),), layout=(256,), names=("d0",)) as scope:
            tile = T.tensor_view(
                T.ptr_of(table[0:0 + 248320, 0:0 + 5120]),
                layout=(248320, 5, 256, 4),
                shape=(248320, 5, 256, 4),
            )
            for i in range(0, 4, 1):
                tile_1 = T.tensor_view(
                    T.ptr_of(token_ids[(cta.d0 * 4) + i:(cta.d0 * 4) + i + 1]),
                    layout=(1,),
                    shape=(1,),
                )
                tile_2 = T.tensor_view(T.ptr_of(tile_1[0:0 + 1]), layout=(), shape=())
                tile_3 = T.tensor_view(
                    T.ptr_of(tile[tile_2:tile_2 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                    layout=(1, 5, 256, 4),
                    shape=(1, 5, 256, 4),
                )
                with scope as threads:
                    dst_frame = T.tensor_view(
                        T.ptr_of(row[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                        layout=(1, 5, 256 @ threads.d0, 4),
                        shape=(1, 5, 256, 4),
                    )
                    T.copy(tile_3, dst_frame)
                    row_view = T.tensor_view(
                        T.ptr_of(row[0:0 + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                        layout=(1, 5, 256 @ threads.d0, 4),
                        shape=(1, 5, 256, 4),
                    )
                    window = T.tensor_view(
                        T.ptr_of(out[(cta.d0 * 4) + i:(cta.d0 * 4) + i + 1, 0:0 + 5, 0:0 + 256, 0:0 + 4]),
                        layout=(1, 5, 256, 4),
                        shape=(1, 5, 256, 4),
                    )
                    T.copy(row_view, window)
