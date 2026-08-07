"""Immutable service descriptors selected by concrete Target values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tilefoundry.ir.core import IRMetadata

AnalysisCallable = Callable[
    ["Module", "Function", "Target", object | None], None
]


@dataclass(frozen=True)
class Analyzer:
    """One analysis service: its identity, dependencies, and owned Metadata."""

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[IRMetadata], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("an analysis needs a non-empty selector")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(
                f"{self.selector}: duplicate entries in requires {list(self.requires)}"
            )
        if self.selector in self.requires:
            raise ValueError(f"{self.selector}: an analysis cannot require itself")
        for produced in self.produces:
            if not isinstance(produced, type) or not issubclass(produced, IRMetadata):
                raise ValueError(
                    f"{self.selector}: produces must name IRMetadata subclasses, "
                    f"got {produced!r}"
                )
        if len(set(self.produces)) != len(self.produces):
            raise ValueError(
                f"{self.selector}: the same Metadata type is produced twice"
            )


ScheduleCallable = Callable[
    ["Module", "Function", "Target", "Topology", object | None],
    "SchedulePlan",
]


@dataclass(frozen=True)
class Scheduler:
    """One scheduler service: the level it solves for, and the solve."""

    topology: str
    solve: ScheduleCallable

    def __post_init__(self) -> None:
        if not self.topology:
            raise ValueError("a schedule algorithm needs a non-empty topology name")


@dataclass(frozen=True)
class CodeGenerator:
    """One immutable Target-owned code-generation service."""

    emit: Callable[
        ["Module", tuple["PrimFunction", ...], "Target"], "LinkableModule"
    ]


__all__ = [
    "AnalysisCallable",
    "Analyzer",
    "CodeGenerator",
    "ScheduleCallable",
    "Scheduler",
]
