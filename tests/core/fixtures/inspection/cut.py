from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="cut", target=cpu.Target())
class Cut:
    @func
    def cut(x: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):
        return tf.slice(x, (start, 0), sizes=(2, 4), strides=(1, 1))
