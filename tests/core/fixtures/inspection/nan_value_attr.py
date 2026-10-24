import math

from tilefoundry.dsl import *


@func
def nan_value(x: Tensor[(8,), "f32"]):
    return tf.full_like(x, value=math.nan)
