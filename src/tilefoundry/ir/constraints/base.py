"""Stage-neutral schedule constraint metadata and source locations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tilefoundry.ir.core.metadata import IRMetadata


@dataclass(frozen=True)
class SourceLocation:
    """Source position associated with an authored scheduling annotation."""

    filename: str = "<string>"
    line: int = 0
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    def describe(self) -> str:
        """Return a compact source position for diagnostics."""
        return f"{self.filename}:{self.line}:{self.column}"


class ConstraintProvenance(Enum):
    """Source category for a schedule constraint."""

    AUTHOR = "author"


@dataclass(frozen=True)
class ScheduleConstraint:
    """Base value for one stage-neutral hard constraint."""

    source_loc: SourceLocation = field(default_factory=SourceLocation)
    provenance: ConstraintProvenance = ConstraintProvenance.AUTHOR


@dataclass(frozen=True)
class ScheduleConstraintMetadata(IRMetadata):
    """Aggregate hard constraints attached to one concrete tensor Expr."""

    constraints: tuple[ScheduleConstraint, ...] = ()
    source_loc: SourceLocation = field(default_factory=SourceLocation)

    def __post_init__(self) -> None:
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError(
                f"schedule constraints at {self.source_loc.describe()} cannot be empty"
            )
        if any(not isinstance(item, ScheduleConstraint) for item in constraints):
            bad = next(item for item in constraints if not isinstance(item, ScheduleConstraint))
            raise TypeError(
                f"schedule constraint metadata expects ScheduleConstraint values, "
                f"got {type(bad).__name__}"
            )
        object.__setattr__(self, "constraints", constraints)


def constraint_metadata(expr: Any) -> ScheduleConstraintMetadata | None:
    """Return schedule metadata attached to ``expr``, if present."""
    for item in getattr(expr, "metadata", ()):
        if type(item) is ScheduleConstraintMetadata:
            return item
    return None


__all__ = [
    "ConstraintProvenance",
    "ScheduleConstraint",
    "ScheduleConstraintMetadata",
    "SourceLocation",
    "constraint_metadata",
]
