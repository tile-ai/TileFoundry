"""CPU compilation target implementation."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from tilefoundry import utils

from . import target
from .registry import register


@register
@dataclass(frozen=True)
class Target(target.Target):
    """Identify the CPU host backend."""

    name = "cpu"

    @typing.override
    def to_python(self) -> utils.PythonExpr:
        """Render this Target as ``cpu.Target()``."""
        if type(self) is not Target:
            return super().to_python()
        return utils.PythonExpr(("from tilefoundry.target import cpu",), "cpu.Target()")


__all__ = ["Target"]
