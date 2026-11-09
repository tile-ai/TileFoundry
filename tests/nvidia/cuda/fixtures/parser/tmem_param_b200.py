"""A tensor-memory parameter on a Blackwell target, which has tensor memory."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@func(target=cuda.Target("nvidia.b200_sxm"), topologies=(Topology("cta", 1),))
def blackwell(x: Tensor[(4,), "f32", None, "tmem"]):
    return x
