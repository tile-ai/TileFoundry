from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class StagedNested:
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
                            layout=(1, 8),
                            shape=(1, 8),
                        )
                        value_1 = T.tensor_view(
                            T.ptr_of(ring_stages[((i * 3) + j) % 2]), layout=(1, 8), shape=(1, 8)
                        )
                        T.copy_async(value, value_1)
                        value_2 = T.tensor_view(
                            T.ptr_of(out[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                            layout=(1, 8),
                            shape=(1, 8),
                        )
                        T.copy(value_1, value_2)
