from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(topologies=(Topology("cta", 132), object()), target=cuda.Target("nvidia.h200_sxm"))
class Invalid:
    pass
