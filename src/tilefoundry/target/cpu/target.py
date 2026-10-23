"""CPU compilation target implementation."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.target.base import Target, register_target
from tilefoundry.utils.python_source import PythonExpr


@register_target
@dataclass(frozen=True)
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name = "cpu"

    def to_python(self) -> PythonExpr:
        if type(self) is not CpuTarget:
            return super().to_python()
        return PythonExpr(("from tilefoundry.target import CpuTarget",), "CpuTarget()")

    def _python_import_module(self) -> str:
        if type(self) is CpuTarget:
            return "tilefoundry.target"
        return super()._python_import_module()



__all__ = ["CpuTarget"]
