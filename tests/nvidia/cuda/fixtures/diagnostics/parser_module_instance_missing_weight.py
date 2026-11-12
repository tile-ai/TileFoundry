from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="main", target=cuda.Target("nvidia.h200_sxm"))
class Invalid:
    @func
    def helper(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return x + w

    @func
    def main(x: Tensor[(4,), "f32"], i: Tensor[(), "i32"]):
        return helper(x)
