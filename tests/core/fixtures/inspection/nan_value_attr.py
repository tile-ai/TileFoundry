import math

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="nan_value", target=CpuTarget())
class NanValue:
    @func
    def nan_value(x: Tensor[(8,), "f32"]):
        return tf.full_like(x, value=math.nan)
