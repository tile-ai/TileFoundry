# analysis target=nvidia.h200_sxm module=CtaSubmeshThreadScope function=gemm topology=cta wave=4/4
# selection requested=memory executed=memory
# memory traffic=gmem:r512B/w512B@logical,r1.00KB/w1.00KB@total,r512B/w512B@cta,r4B/w4B@thread;rmem:r512B/w512B@logical,r1.00KB/w1.00KB@total,r512B/w512B@cta,r4B/w4B@thread footprint=v2:15:512B;x:512B footprint-precision=exact peak=gmem:1.00KB;rmem:512B persistent=gmem:512B
#   buffer=x holds=1.00KB time=none space=cta.b reuse=512B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(4, 32), "f32"], out: Tensor[(4, 32), "f32"]):
    with Mesh((Topology("cta", 4),), layout=(2, 2), names=("d0", "d1")) as cta:
        with cta[:, :1] as scope:
            with Mesh(
                (Topology("thread", 128),), layout=(4, 32), names=("d0", "d1")
            ) as scope_1:
                value = T.alloc_tensor(
                    tensor_type=Tensor[
                        (4, 32), "f32", (4 @ scope_1.d0, 32 @ scope_1.d1), "rmem"
                    ]
                )
                value_1 = T.tensor_view(T.ptr_of(x), layout=(4, 32), shape=(4, 32))
                value_2 = T.tensor_view(
                    T.ptr_of(value), layout=(4 @ scope_1.d0, 32 @ scope_1.d1), shape=(4, 32)
                )
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(out), layout=(4, 32), shape=(4, 32))
                T.copy(value_2, value_3)
