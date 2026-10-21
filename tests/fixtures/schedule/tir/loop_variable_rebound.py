from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32", (2, 4), "rmem"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            value = T.tensor_view(T.ptr_of(x), layout=(2, 4), shape=(2, 4))
            value_1 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            T.copy(value, value_1)
            for u in range(0, 2, 1):
                value_2 = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", (2, 4), "rmem"])
                value_3 = T.tensor_view(T.ptr_of(x), layout=(2, 4), shape=(2, 4))
                value_4 = T.tensor_view(T.ptr_of(value_2), layout=(2, 4), shape=(2, 4))
                T.copy(value_3, value_4)
                value_5 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                value_6 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                T.binary(value_5, value_4, value_6, kind=BinaryKind.ADD)
