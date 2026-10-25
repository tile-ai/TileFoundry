import math

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="nan_value", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 1),))
class NanValue:
    @prim_func
    def nan_value(x: Tensor[(8,), "f32", "rmem"]):
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as thread:  # noqa: F841
            T.fill(x, math.nan)
