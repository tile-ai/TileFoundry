from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32),),
)
class LoopCarryWindow:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def gemm(x: Tensor[(4, 4), "f32"], out: Tensor[(4, 4), "f32"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                T.fill(out, 0.0)
                out_1 = T.alloc_tensor(tensor_type=Tensor[(4, 4), "f32", "rmem"])
                T.fill(out_1, 0.0)
                for m in range(0, 2, 1):
                    value = T.tensor_view(
                        T.ptr_of(out_1[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                    )
                    T.fill(value, 0.0)
                    for k in range(0, 3, 1):
                        value_1 = T.alloc_tensor(tensor_type=Tensor[(2, 4), "f32", "rmem"])
                        value_2 = T.tensor_view(
                            T.ptr_of(x[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                        )
                        value_3 = T.tensor_view(T.ptr_of(value_1), layout=(2, 4), shape=(2, 4))
                        T.copy(value_2, value_3)
                        value_4 = T.tensor_view(
                            T.ptr_of(out_1[2 * m:2 * m + 1, 0:0 + 1]), layout=(2, 4), shape=(2, 4)
                        )
                        T.binary(value_4, value_3, value_4, kind=BinaryKind.ADD)
                value_5 = T.tensor_view(T.ptr_of(out_1), layout=(4, 4), shape=(4, 4))
                value_6 = T.tensor_view(T.ptr_of(out), layout=(4, 4), shape=(4, 4))
                window = T.tensor_view(
                    T.ptr_of(value_6[0:0 + 4, 0:0 + 4]), layout=(4, 4), shape=(4, 4)
                )
                T.copy(value_5, window)
