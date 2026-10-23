from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_cuda import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(
    x: Tensor[(4, 4), "bf16"], idx: Tensor[(2,), "i64"], out: Tensor[(2, 4), "bf16", "smem"]
):
    with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
            T.fill(out, 0.0)
            for r in range(0, 2, 1):
                value = T.tensor_view(T.ptr_of(x), layout=(4, 4), shape=(4, 4))
                source = T.tensor_view(T.ptr_of(value), layout=(4, 4), shape=(4, 4))
                row_dst = T.tensor_view(
                    16, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                )
                value_1 = T.tensor_view(T.ptr_of(row_dst), layout=(1, 4), shape=(1, 4))
                value_2 = T.tensor_view(T.ptr_of(idx[r:r + 1]), layout=(1,), shape=(1,))
                T.copy_async(source, value_1, value_2, fill=0)
                with scope[:] as scope_1:
                    row_dst_2 = T.tensor_view(
                        16, dtype="bf16", storage="smem", layout=(1, 4), shape=(1, 4)
                    )
                    value_3 = T.tensor_view(T.ptr_of(row_dst_2), layout=(1, 4), shape=(1, 4))
                    value_4 = T.tensor_view(T.ptr_of(out), layout=(2, 4), shape=(2, 4))
                    window = T.tensor_view(
                        T.ptr_of(value_4[r:r + 1, 0:0 + 4]), layout=(1, 4), shape=(1, 4)
                    )
                    T.copy(value_3, window)
