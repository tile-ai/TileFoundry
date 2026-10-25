"""A shared-memory parameter on a target that has shared memory."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def fine(x: Tensor[(4,), "f32", None, "smem"]):
    return x
