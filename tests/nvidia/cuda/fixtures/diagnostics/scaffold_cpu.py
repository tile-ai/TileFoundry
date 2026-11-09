from tilefoundry.dsl import *
from tilefoundry.target import cpu

TARGET = cpu.Target()


@module(entry="run", target=TARGET)
class Root:
    @prim_func(target=TARGET)
    def run(x: Tensor[(8,), "f32"]):
        pass
