from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(topologies=[Topology("cta", 132)], target=CudaTarget("nvidia.h200_sxm"))
class Invalid:
    pass
