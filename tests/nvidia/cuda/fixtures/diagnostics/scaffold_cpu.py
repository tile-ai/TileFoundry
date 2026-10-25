from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

TARGET = CpuTarget()


@module(entry="run", target=TARGET)
class Root:
    @prim_func(target=TARGET)
    def run(x: Tensor[(8,), "f32"]):
        pass
