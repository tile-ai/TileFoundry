"""A matmul whose b_layout literal names no layout."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="invalid_layout", target=CpuTarget())
class MatmulBadLayout:
    @func
    def invalid_layout(
        a: Tensor[(2, 3), "f32"], b: Tensor[(3, 4), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.matmul(a, b, b_layout="bad")
