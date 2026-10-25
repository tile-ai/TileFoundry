from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(target=CudaTarget("nvidia.h200_sxm"))
class Model:
    @func(mesh=Mesh((Topology("thread", 8),), layout=(8,), names=("lane",)))
    def f(x: Tensor[(8, 16), "f32", (8, 16) @ {B(), mesh.lane @ P("sum")}, "rmem"]):
        return x
