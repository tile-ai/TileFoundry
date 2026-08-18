"""Immutable service descriptors selected by concrete Target values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tilefoundry.ir.core import IRMetadata

AnalysisCallable = Callable[
    ["Module", "Function", "Target", str | None, object | None], None
]


@runtime_checkable
class AnalysisInputChecker(Protocol):
    """What one analysis needs of a program before any analysis writes.

    The three hooks are the three questions that can be asked without reading a
    record: what the target must state, what each call must carry, and what the
    function as a whole must hold. A checker states requirements; it never
    attaches Metadata, and never stands in for what an analysis concludes.
    """

    def check_target(self, ctx: "AnalysisCheckContext") -> None: ...

    def check_call(self, call: "Call", ctx: "AnalysisCheckContext") -> None: ...

    def finish(self, function: "Function", ctx: "AnalysisCheckContext") -> None: ...


class _NoInputCheck:
    """The requirement an analysis states by saying nothing."""

    def check_target(self, ctx: "AnalysisCheckContext") -> None:
        return None

    def check_call(self, call: "Call", ctx: "AnalysisCheckContext") -> None:
        return None

    def finish(self, function: "Function", ctx: "AnalysisCheckContext") -> None:
        return None


NO_INPUT_CHECK = _NoInputCheck()


@dataclass(frozen=True)
class Analyzer:
    """One analysis service: its identity, dependencies, and owned Metadata."""

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[IRMetadata], ...] = field(default=())
    input_checker: AnalysisInputChecker = NO_INPUT_CHECK

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
    "NO_INPUT_CHECK",
    "AnalysisCallable",
    "AnalysisInputChecker",
    "Analyzer",
    "CodeGenerator",
    "ScheduleCallable",
    "Scheduler",
]
