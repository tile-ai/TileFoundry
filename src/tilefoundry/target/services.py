"""Immutable service descriptors selected by concrete Target values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tilefoundry.ir.core import IRMetadata

AnalysisCallable = Callable[
    ["Function", "AnalyzeContext"], None
]


@runtime_checkable
class AnalysisChecker(Protocol):
    """What one analysis needs of a program before any analysis writes.

    A checker states requirements; it never attaches Metadata, and never stands
    in for what an analysis concludes. It asks target-wide questions once and
    visits the program for any occurrence-wide requirements of its own.
    """

    def check_target_facts(self, ctx: "AnalysisCheckContext") -> None: ...

    def visit(self, expr: "Expr", ctx: "AnalysisCheckContext") -> None: ...


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

    def get_checker(self) -> AnalysisChecker | None:
        """Return this analysis's program requirements, if it states any."""
        return None


@dataclass(frozen=True)
class CodeGenerator:
    """One immutable Target-owned code-generation service."""

    emit: Callable[
        ["Module", tuple["PrimFunction", ...], "Target"], "LinkableModule"
    ]


__all__ = [
    "AnalysisCallable",
    "AnalysisChecker",
    "Analyzer",
    "CodeGenerator",
]
