from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.hir.function import Function

from .constraints import ConstraintList


@dataclass(frozen=True, slots=True)
class ScheduleInput:
    function: Function
    constraints: ConstraintList

    def __post_init__(self) -> None:
        if not isinstance(self.function, Function):
            raise TypeError("ScheduleInput.function must be a hir.Function")
        if not isinstance(self.constraints, ConstraintList):
            raise TypeError("ScheduleInput.constraints must be a ConstraintList")


__all__ = ["ScheduleInput"]
