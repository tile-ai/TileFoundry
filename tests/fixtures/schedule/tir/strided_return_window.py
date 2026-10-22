from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def gemm(x: Tensor[(32, 14336), "bf16"], out: Tensor[(24, 256), "bf16", "rmem"]):
    with Mesh((Topology("cta", 32),), layout=(32,), names=("d0",)) as cta:
        with Mesh((Topology("thread", 1),), layout=(1,), names=("d0",)) as scope:
            value = T.tensor_view(
                T.ptr_of(x[cta.d0:cta.d0 + 1, 0:0 + 1]),
                layout=((24, 256), (512, 1)),
                shape=(24, 256),
            )
            value_1 = T.tensor_view(T.ptr_of(out), layout=(24, 256), shape=(24, 256))
            T.copy(value, value_1)
