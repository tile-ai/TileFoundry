"""A matmul with a misspelled attribute keyword."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="refused", target=CpuTarget())
class MatmulMisspelledAttribute:
    @func
    def refused(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
        return tf.matmul(a, b, a_layoutt="MK")
