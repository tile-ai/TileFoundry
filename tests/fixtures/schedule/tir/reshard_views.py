from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "f32"], out: Tensor[(8,), "f32", (8,), "rmem"]):
    with Mesh((Topology("cta", 2),), layout=(2,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            value = T.tensor_view(T.ptr_of(x), layout=(8,), shape=(8,))
            value_1 = T.tensor_view(T.ptr_of(out), layout=(8,), shape=(8,))
            T.copy(value, value_1)
