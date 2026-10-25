from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(topologies=(Topology("cta", 132), object()), target=CudaTarget("nvidia.h200_sxm"))
class Invalid:
    pass
