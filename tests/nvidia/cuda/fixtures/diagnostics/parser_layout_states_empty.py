from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(target=cuda.Target("nvidia.h200_sxm"))
class Model:
    mesh = Mesh((Topology("thread", 8),), layout=(8,), names=("lane",))

    @func
    def f(x: Tensor[(8, 16), "f32", (8, 16) @ {}, "rmem"]):
        with mesh:
            return x
