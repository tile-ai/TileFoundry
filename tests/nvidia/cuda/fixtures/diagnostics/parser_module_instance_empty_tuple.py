from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"))
class Invalid:
    layers = ModuleList(())

    @func
    def main(x: Tensor[(4,), "f32"], i: Tensor[(), "i32"]):
        return x
