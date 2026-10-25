from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class LoopCarryFromParam:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(2, 4), "bf16"], y: Tensor[(1, 4), "bf16"], out: Tensor[(2, 4), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                value = T.tensor_view(T.ptr_of(x), layout=(2, 4), shape=(2, 4))
                value_1 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                T.copy(value, value_1)
                for r in range(0, 2, 1):
                    value_2 = T.tensor_view(T.ptr_of(y), layout=(1, 4), shape=(1, 4))
                    value_3 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                    window = T.tensor_view(
                        T.ptr_of(value_3[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy(value_2, window)
