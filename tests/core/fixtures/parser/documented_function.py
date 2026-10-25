"""A function whose leading docstring is documentation, not a statement."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="documented", target=CpuTarget())
class Documented:
    @func
    def documented(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        """This is documentation, not an HIR statement."""
        return x
