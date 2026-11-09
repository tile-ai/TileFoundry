"""Stage-neutral where-clause metadata and source locations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tilefoundry.ir import expr


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


class Provenance(Enum):
    """Source category for a where clause."""

    AUTHOR = "author"


@dataclass(frozen=True)
class Where:
    """Base value for one stage-neutral hard constraint."""

    source_loc: SourceLocation = field(default_factory=SourceLocation)
    provenance: Provenance = Provenance.AUTHOR


@dataclass(frozen=True)
class WhereMetadata(expr.IRMetadata):
    """Aggregate hard constraints attached to one concrete tensor Expr."""

    constraints: tuple[Where, ...] = ()
    source_loc: SourceLocation = field(default_factory=SourceLocation)

    def __post_init__(self) -> None:
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError(f"where clauses at {self.source_loc.describe()} cannot be empty")
        if any(not isinstance(item, Where) for item in constraints):
            bad = next(item for item in constraints if not isinstance(item, Where))
            raise TypeError(
                f"where-clause metadata expects Where values, got {type(bad).__name__}"
            )
        object.__setattr__(self, "constraints", constraints)


def metadata(node: Any) -> WhereMetadata | None:
    """Return schedule metadata attached to ``expr``, if present."""
    for item in getattr(node, "metadata", ()):
        if type(item) is WhereMetadata:
            return item
    return None


__all__ = ["Provenance", "SourceLocation", "Where", "WhereMetadata", "metadata"]
