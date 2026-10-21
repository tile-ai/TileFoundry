from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(1, 4), "f32"], out: Tensor[(2, 4), "f32"]):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            for _i in range(0, 0, 1):
                value = T.alloc_tensor(tensor_type=Tensor[(1, 4), "f32", (1, 4), "rmem"])
                value_1 = T.tensor_view(T.ptr_of(x), layout=(1, 4), shape=(1, 4))
                value_2 = T.tensor_view(T.ptr_of(value), layout=(1, 4), shape=(1, 4))
                T.copy(value_1, value_2)
                value_3 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                window = T.tensor_view(
                    T.ptr_of(value_3[0:0 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_2, window)
