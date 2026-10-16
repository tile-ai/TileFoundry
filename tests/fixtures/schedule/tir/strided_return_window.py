# analysis target=nvidia.h200_sxm module=StridedReturnWindow function=gemm topology=cta wave=32/32
# selection requested=memory executed=memory
# memory traffic=gmem:r12.00KB/w0@logical,r384.00KB/w0@total,r12.00KB/w0@cta,r12.00KB/w0@thread;rmem:r32B/w12.00KB@logical,r1.00KB/w384.00KB@total,r32B/w12.00KB@cta,r32B/w12.00KB@thread footprint=x:384.00KB footprint-precision=exact peak=gmem:896.00KB;rmem:12.00KB persistent=gmem:896.00KB

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32, 14336), "bf16"], out: Tensor[(24, 256), "bf16", (24, 256), "rmem"]):
    with Mesh((Topology("cta", 32),), layout=(32,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            value = T.tensor_view(
                T.ptr_of(x[cta.d0:cta.d0 + 1, 0:0 + 1]),
                layout=((24, 256), (512, 1)),
                shape=(24, 256),
            )
            value_1 = T.tensor_view(T.ptr_of(out), layout=(24, 256), shape=(24, 256))
            T.copy(value, value_1)
