import math

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="nan_value", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("thread", 1),))
class NanValue:
    @prim_func
    def nan_value(x: Tensor[(8,), "f32", "rmem"]):
        with Mesh((Topology("thread", 1),), layout=(1,), names=("t",)) as thread:
            T.fill(x, math.nan)
