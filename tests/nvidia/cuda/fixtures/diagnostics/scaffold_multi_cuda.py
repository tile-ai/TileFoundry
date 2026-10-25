from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

TARGET = CudaTarget("nvidia.h200_sxm", device_count=2)


@module(entry="run", target=TARGET)
class Root:
    @prim_func(target=TARGET)
    def run(x: Tensor[(8,), "f32"]):
        pass
