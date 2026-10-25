"""A literal index written with true division."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="stage", target=CpuTarget())
class LiteralIndexDivision:
    @func
    def stage(x: Tensor[(1, 256, 64), "f32"]):
        return x[:, 8 / 2, :]
