"""A CPU host ``@prim_func`` that launches a CUDA kernel defined above it in the class body."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget


@module(entry="host", target=CudaTarget("nvidia.h200_sxm"))
class _Launch:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def dev(a: Tensor[(8,), "f32"]):  # noqa: ARG001
        with Mesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as m:
            T.sync(m)

    @prim_func(target=CpuTarget())
    def host(a: Tensor[(8,), "f32"]):
        launch(dev, a, grid=(1, 1, 1), block=(8, 1, 1))  # noqa: F821
