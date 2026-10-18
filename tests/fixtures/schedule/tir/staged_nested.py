# analysis target=nvidia.h200_sxm module=StagedNested function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r96B/w0@logical,r96B/w0@total,r96B/w0@cta,r96B/w0@thread;rmem:r416B/w288B@logical,r480B/w288B@total,r480B/w288B@cta,r480B/w288B@thread;smem:r96B/w96B@logical,r96B/w96B@total,r96B/w96B@cta,r96B/w96B@thread footprint=x:16B footprint-precision=exact peak=gmem:96B;rmem:96B;smem:32B persistent=gmem:96B

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(6, 8), "bf16"], out: Tensor[(6, 8), "bf16", "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                T.tensor_view(16, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
            )
            for i in range(0, 2, 1):
                for j in range(0, 3, 1):
                    value = T.tensor_view(
                        T.ptr_of(x[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                        layout=((1, 8), (0, 1)),
                        shape=(1, 8),
                    )
                    value_1 = T.tensor_view(
                        T.ptr_of(ring_stages[((i * 3) + j) % 2]), layout=(1, 8), shape=(1, 8)
                    )
                    T.copy_async(value, value_1)
                    value_2 = T.tensor_view(
                        T.ptr_of(out[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                        layout=((1, 8), (0, 1)),
                        shape=(1, 8),
                    )
                    T.copy(value_1, value_2)
