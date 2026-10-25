from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

N = DimVar("n_check", 1, 8)


@module(entry="run")
class Leaf:
    @func
    def run(x: Tensor[(N,), "f32"], w: ConstTensor[(1,), "f32"]) -> Tensor[(N,), "f32"]:
        return tf.mul(x, w)


@module(entry="root", target=CudaTarget("nvidia.h200_sxm"))
class Composed:
    leaf = Leaf

    @func
    def root(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        return leaf(x)
