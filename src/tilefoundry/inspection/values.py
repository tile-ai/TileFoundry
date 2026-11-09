"""Canonical metadata comment printer."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr


class Prose(str):
    """Sentence text rendered as a quoted DSL string."""


@dataclass(frozen=True)
class ReportIdentity(expr.IRMetadata):
    target: str = ""
    module: str = ""
    function: str = ""
    topology: str = "none"


@dataclass(frozen=True)
class ReportSelection(expr.IRMetadata):
    requested: tuple[str, ...] = ()
    executed: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdvisorySummary(expr.IRMetadata):
    text: Prose


@dataclass(frozen=True)
class ErrorSummary(expr.IRMetadata):
    text: Prose


@dataclass(frozen=True)
class PerformanceSummaryView(expr.IRMetadata):
    root: str = ""
    predicted_ns: int = 0
    wave: str = ""


__all__ = [
    "Prose",
    "ReportIdentity",
    "ReportSelection",
    "AdvisorySummary",
    "ErrorSummary",
    "PerformanceSummaryView",
]
