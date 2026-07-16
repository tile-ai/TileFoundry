from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tilefoundry.ir.core.metadata import IRMetadata


@dataclass(frozen=True)
class SourceLocation:
    filename: str = "<string>"
    line: int = 0
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None


class ConstraintProvenance(Enum):
    AUTHOR = "author"


@dataclass(frozen=True)
class AgentConstraint:
    source_loc: SourceLocation = field(default_factory=SourceLocation)
    provenance: ConstraintProvenance = ConstraintProvenance.AUTHOR


@dataclass(frozen=True)
class AgentConstraintsMetadata(IRMetadata):
    constraints: tuple[AgentConstraint, ...] = ()
    source_loc: SourceLocation = field(default_factory=SourceLocation)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))


def merge_constraints(
    metadata: tuple[IRMetadata, ...],
    constraints: tuple[AgentConstraint, ...],
    source_loc: SourceLocation,
) -> tuple[IRMetadata, ...]:
    """Replace the expression's constraint metadata with one merged record."""
    prior: list[AgentConstraint] = []
    other: list[IRMetadata] = []
    for item in metadata:
        if isinstance(item, AgentConstraintsMetadata):
            prior.extend(item.constraints)
        else:
            other.append(item)
    other.append(
        AgentConstraintsMetadata(
            constraints=tuple((*prior, *constraints)), source_loc=source_loc
        )
    )
    return tuple(other)


def constraint_metadata(expr: Any) -> AgentConstraintsMetadata | None:
    for item in getattr(expr, "metadata", ()):
        if isinstance(item, AgentConstraintsMetadata):
            return item
    return None


__all__ = [
    "AgentConstraint",
    "AgentConstraintsMetadata",
    "ConstraintProvenance",
    "SourceLocation",
    "constraint_metadata",
    "merge_constraints",
]
