# analysis target=nvidia.h200_sxm module=LoopCarryWindow function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r64B/w128B@logical,r192B/w128B@total,r192B/w128B@cta,r192B/w128B@thread;rmem:r624B/w416B@logical,r688B/w576B@total,r688B/w576B@cta,r688B/w576B@thread footprint=v10:29:64B;x:32B footprint-precision=exact peak=gmem:128B;rmem:96B persistent=gmem:64B
#   buffer=x holds=32B time=k space=none reuse=64B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(4, 4), "f32"], out: Tensor[(4, 4), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("block",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("lane",)) as scope:
            value = T.tensor_view(T.ptr_of(out), layout=(4, 4), shape=(4, 4))
            T.fill(value, 0.0)
            out_1 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
            T.fill(out_1, 0.0)
            for m in range(0, 2, 1):
                value_1 = T.tensor_view(T.ptr_of(out_1), layout=(4, 4), shape=(4, 4))
                value_2 = T.tensor_view(T.ptr_of(value_1), layout=(4, 4), shape=(4, 4))
                value_3 = T.tensor_view(
                    T.ptr_of(value_2[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                )
                T.fill(value_3, 0.0)
                for k in range(0, 3, 1):
                    value_4 = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", "rmem"])
                    value_5 = T.tensor_view(T.ptr_of(x), layout=(4, 4), shape=(4, 4))
                    value_6 = T.tensor_view(
                        T.ptr_of(value_5[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                    )
                    value_7 = T.tensor_view(T.ptr_of(value_4), layout=(2, 4), shape=(2, 4))
                    T.copy(value_6, value_7)
                    value_8 = T.tensor_view(T.ptr_of(out_1), layout=(4, 4), shape=(4, 4))
                    value_9 = T.tensor_view(T.ptr_of(value_8), layout=(4, 4), shape=(4, 4))
                    value_10 = T.tensor_view(
                        T.ptr_of(value_9[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                    )
                    value_11 = T.tensor_view(T.ptr_of(value_10), layout=(2, 4), shape=(2, 4))
                    value_12 = T.tensor_view(T.ptr_of(value_11), layout=(2, 4), shape=(2, 4))
                    T.binary(value_11, value_7, value_12, kind=BinaryKind.ADD)
            value_13 = T.tensor_view(T.ptr_of(out_1), layout=(4, 4), shape=(4, 4))
            value_14 = T.tensor_view(T.ptr_of(value_13), layout=(4, 4), shape=(4, 4))
            value_15 = T.tensor_view(T.ptr_of(out), layout=(4, 4), shape=(4, 4))
            window = T.tensor_view(
                T.ptr_of(value_15[0:0 + 4, 0:0 + 4]), layout=(4, 4), shape=(4, 4)
            )
            T.copy(value_14, window)
