"""A Module whose entry names no collected function."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="absent", target=CpuTarget())
class _WrongEntry:
    @func
    def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)
