# analysis target=nvidia.h200_sxm module=StagedDynamicInner function=gemm topology=cta wave=1/1
# selection requested=memory executed=memory
# memory traffic=gmem:r16B/w0@logical,r992B/w0@total,r992B/w0@cta,r992B/w0@thread;rmem:r992B/w528B@logical,r1.94KB/w1.95KB@total,r1.94KB/w1.95KB@cta,r1.94KB/w1.95KB@thread;smem:r16B/w16B@logical,r992B/w992B@total,r992B/w992B@cta,r992B/w992B@thread footprint=x:16B footprint-precision=exact peak=gmem:16B;rmem:16B;smem:32B persistent=gmem:16B
#   buffer=x holds=16B time=i space=none reuse=16B fits=yes precision=exact

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(1, 8), "bf16"], out: Tensor[(1, 8), "bf16", "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            ring_stages = (
                T.tensor_view(0, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                T.tensor_view(16, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
            )
            stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_counter, 0)
            stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
            T.fill(stage_one, 1)
            for i in range(0, 2, 1):
                for j in range(0, scope.d0, 1):
                    value = T.tensor_view(T.ptr_of(x), layout=(1, 8), shape=(1, 8))
                    value_1 = T.tensor_view(
                        T.ptr_of(ring_stages[stage_counter % 2]), layout=(1, 8), shape=(1, 8)
                    )
                    T.copy_async(value, value_1)
                    value_2 = T.tensor_view(T.ptr_of(out), layout=(1, 8), shape=(1, 8))
                    T.copy(value_1, value_2)
                    T.binary(stage_counter, stage_one, stage_counter, kind=BinaryKind.ADD)
