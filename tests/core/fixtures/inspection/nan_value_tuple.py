import math

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="nan_value", target=cpu.Target())
class NanValue:
    @func
    def nan_value(x: Tensor[(8,), "f32"]):
        return (x, math.nan)
