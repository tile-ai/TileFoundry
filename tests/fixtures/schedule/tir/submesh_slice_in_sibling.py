from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(8,), "f32"], out: Tensor[(4,), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 64),), layout=(64,), names=("d0",)) as scope:
            with scope[:32] as scope_1:
                value_1 = T.tensor_view(T.ptr_of(x), layout=(8,), shape=(8,))
                value = T.tensor_view(0, dtype="f32", storage="smem", layout=(8,), shape=(8,))
                value_2 = T.tensor_view(T.ptr_of(value), layout=(8,), shape=(8,))
                T.copy(value_1, value_2)
            with scope[32:] as scope_2:
                value_5 = T.tensor_view(0, dtype="f32", storage="smem", layout=(8,), shape=(8,))
                value_3 = T.tensor_view(T.ptr_of(value_5), layout=(4,), shape=(4,))
                value_4 = T.tensor_view(T.ptr_of(out), layout=(4,), shape=(4,))
                T.copy(value_3, value_4)
