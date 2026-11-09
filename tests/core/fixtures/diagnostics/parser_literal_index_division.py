"""A literal index written with true division."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="stage", target=cpu.Target())
class LiteralIndexDivision:
    @func
    def stage(x: Tensor[(1, 256, 64), "f32"]):
        return x[:, 8 / 2, :]
