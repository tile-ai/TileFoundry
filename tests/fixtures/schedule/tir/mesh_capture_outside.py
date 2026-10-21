from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32,), "f32"], scratch: Tensor[(32,), "f32"], out: Tensor[(32,), "f32"]):
    with Mesh((Topology("cta", 2),), layout=(2,), names=("d0",)) as scope:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope_1:
            value_1 = T.tensor_view(T.ptr_of(x), layout=(32,), shape=(32,))
            value_2 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            value_3 = T.tensor_view(T.ptr_of(value_2), layout=(32,), shape=(32,))
            T.copy(value_1, value_3)
    with Mesh((Topology("cta", 2),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope_2:
            value_4 = T.tensor_view(T.ptr_of(scratch[0:0 + 32]), layout=(32,), shape=(32,))
            value_5 = T.tensor_view(T.ptr_of(value_4), layout=(32,), shape=(32,))
            value_6 = T.tensor_view(T.ptr_of(out), layout=(32,), shape=(32,))
            T.copy(value_5, value_6)
