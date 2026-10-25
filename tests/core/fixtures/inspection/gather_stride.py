from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="gather", target=CpuTarget())
class Gather:
    @func
    def gather(x: Tensor[(8,), "f32"]):
        out = tf.zeros(Tensor[(), "f32"])
        for i in range(0, 8, 2):
            out = tf.add(out, x[i])
        return out
