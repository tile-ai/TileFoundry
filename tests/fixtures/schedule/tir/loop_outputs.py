from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"), output_count=3)
def gemm(
    x: Tensor[(2, 4), "bf16"], scratch: Tensor[(8,), "bf16"], out: Tensor[(2, 4), "bf16"], out_1: Tensor[(2, 4), "bf16"], out_2: Tensor[(2, 4), "bf16"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            for row in range(0, 2, 1):
                value = T.tensor_view(
                    T.ptr_of(x[row:row + 1, 0:0 + 1]), layout=(1, 4), shape=(1, 4)
                )
                landing = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                value_1 = T.tensor_view(T.ptr_of(landing), layout=(1, 4), shape=(1, 4))
                T.copy_async(value, value_1)
                value_2 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                window = T.tensor_view(
                    T.ptr_of(value_2[row:row + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_1, window)
            value_3 = T.tensor_view(T.ptr_of(x), layout=(2, 4), shape=(2, 4))
            value_4 = T.tensor_view(T.ptr_of(out_1), layout=(2, 4), shape=(2, 4))
            T.copy(value_3, value_4)
            for row_2 in range(0, 2, 1):
                temporary_1 = T.tensor_view(
                    T.ptr_of(scratch[0:0 + 8]), layout=(2, 4), shape=(2, 4)
                )
                T.fill(temporary_1, 0.0)
                value_5 = T.tensor_view(
                    T.ptr_of(x[row_2:row_2 + 1, 0:0 + 1]), layout=(1, 4), shape=(1, 4)
                )
                landing_1 = T.tensor_view(
                    0, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                value_6 = T.tensor_view(T.ptr_of(landing_1), layout=(1, 4), shape=(1, 4))
                T.copy_async(value_5, value_6)
                value_7 = T.tensor_view(T.ptr_of(temporary_1), layout=(2, 4), shape=(2, 4))
                window_1 = T.tensor_view(
                    T.ptr_of(value_7[row_2:row_2 + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                )
                T.copy(value_6, window_1)
                value_8 = T.tensor_view(T.ptr_of(out_1), layout=(2, 4), shape=(2, 4))
                T.copy(value_7, value_8)
            value_9 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
            value_10 = T.tensor_view(T.ptr_of(out_2), layout=(2, 4), shape=(2, 4))
            T.copy(value_9, value_10)
