"""A device root must not load the CPU host supplied by its child."""

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="host", target=CUDA, topologies=(Topology("thread", 32),))
class Child:
    @prim_func(target=CUDA)
    def device(x: Tensor[(8,), "f32"]):
        pass

    @prim_func(target=CpuTarget())
    def host(x: Tensor[(8,), "f32"]):
        launch(device, x, grid=(1, 1, 1), block=(32, 1, 1))


@module(entry="device", target=CUDA, topologies=(Topology("thread", 32),))
class NoRootHost:
    child = Child

    @prim_func(target=CUDA)
    def device(x: Tensor[(8,), "f32"]):
        pass
