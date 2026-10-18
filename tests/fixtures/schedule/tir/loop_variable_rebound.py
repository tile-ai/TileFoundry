# analysis target=nvidia.h200_sxm module=LoopVariableRebound function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r64B/w0@logical,r96B/w0@total,r96B/w0@cta,r96B/w0@thread;rmem:r208B/w192B@logical,r208B/w224B@total,r208B/w224B@cta,r208B/w224B@thread footprint=x:32B footprint-precision=exact peak=gmem:32B;rmem:64B persistent=gmem:32B
#   buffer=x holds=32B time=u space=none reuse=32B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32", (2, 4), "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("b",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as scope:
            T.fill(out, 0.0)
            value = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            T.copy(x, value)
            for u in range(0, 2, 1):
                value_1 = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", (2, 4), "rmem"])
                T.copy(x, value_1)
                value_2 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                value_3 = T.tensor_view(T.ptr_of(value_2), layout=(2, 4), shape=(2, 4))
                T.binary(value_3, value_1, value_2, kind=BinaryKind.ADD)
