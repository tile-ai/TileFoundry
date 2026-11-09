"""A matmul that binds a_layout positionally and by keyword."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="refused", target=cpu.Target())
class MatmulAttributeTwice:
    @func
    def refused(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
        return tf.matmul(a, b, "MK", a_layout="MK")
