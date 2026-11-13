"""Host launches of two instance definitions chosen by constant index."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

CUDA = cuda.Target("nvidia.h200_sxm")


@module(entry="run", topologies=(Topology("thread", 32),))
class Scale:
    @prim_func(target=CUDA)
    def run(x: Tensor[(8,), "f32"]):
        pass


@module(entry="run", topologies=(Topology("thread", 64),))
class Shift:
    @prim_func(target=CUDA)
    def run(x: Tensor[(8,), "f32"]):
        pass


@module(entry="host", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    layers = ModuleList((Scale, Shift))

    @prim_func(target=cpu.Target())
    def host(x: Tensor[(8,), "f32"]):
        launch(layers[0].run, x, grid=(1, 1, 1), block=(32, 1, 1))
        launch(layers[1].run, x, grid=(1, 1, 1), block=(64, 1, 1))
