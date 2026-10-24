from __future__ import annotations

from tilefoundry.dsl import *


@func
def sh(x: Tensor[(4, 8), "f32"]):
    vals, idx = tf.topk(x, k=3, axis=-1, largest=True, sorted=True)
    return vals
