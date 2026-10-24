from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_N = DimVar("n_print", 1, 8)

@module(entry="run")
class _WeightedAtAnySize:
    @func
    def run(x: Tensor[(_N, 8), "f32"], w: ConstTensor[(8, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        return tf.matmul(x, w)


@module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
class _Dispatching:
    leaf = _WeightedAtAnySize

    @func
    def dispatch(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        pass

    @dispatch.specialize(RangePattern("n_print", 1, 8))
    def child_dispatch(x: Tensor[(_N, 8), "f32"]) -> Tensor[(_N, 8), "f32"]:
        return leaf(x)  # noqa: F821
