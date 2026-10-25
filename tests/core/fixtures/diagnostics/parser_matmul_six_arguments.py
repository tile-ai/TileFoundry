"""A matmul given six positional arguments."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="refused", target=CpuTarget())
class MatmulSixArguments:
    @func
    def refused(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
        return tf.matmul(a, b, "MK", "KN", "f32", "KN")
