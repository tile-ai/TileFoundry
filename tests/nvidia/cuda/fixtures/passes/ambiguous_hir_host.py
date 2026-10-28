"""A CPU root launching two HIR functions has no single device root to lower."""

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(entry="host", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 32),))
class AmbiguousHIRHost:
    @func
    def first(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return x

    @func
    def second(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return x

    @prim_func(target=CpuTarget())
    def host(x: Tensor[(8,), "f32"], first_out: Tensor[(8,), "f32"], second_out: Tensor[(8,), "f32"]):
        launch(first, x, first_out, grid=(1, 1, 1), block=(32, 1, 1))
        launch(second, x, second_out, grid=(1, 1, 1), block=(32, 1, 1))
