from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

TARGET = CudaTarget("nvidia.h200_sxm", device_count=2)


@module(entry="main", target=TARGET)
class Root:
    @prim_func(target=TARGET)
    def run(x: Tensor[(8,), "f32"]):
        pass

    @prim_func(target=CpuTarget())
    def main(x: Tensor[(8,), "f32"]):
        launch(run, x, grid=(1, 1, 1), block=(1, 1, 1))
