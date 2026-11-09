"""A dimension index written with true division."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

N = DimVar("N", 1, 256)


@module(entry="stage", target=cpu.Target())
class SymbolicIndexDivision:
    @func
    def stage(x: Tensor[(1, 256, 64), "f32"]):
        return x[:, N / 2, :]
