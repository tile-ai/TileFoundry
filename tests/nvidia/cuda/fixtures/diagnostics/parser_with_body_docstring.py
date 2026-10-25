from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def nested(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    with Mesh(("cta",), layout=(1,), names=("unit",)) as _mesh:
        """A string in a with body remains an ordinary statement."""
        return x
