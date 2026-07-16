from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.ir.core import Expr
from tilefoundry.ir.target.storage import StorageKind


@dataclass(frozen=True, slots=True)
class SourceLocation:
    filename: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None


class ConstraintProvenance(str, Enum):
    AUTHOR = "author"


@dataclass(frozen=True, slots=True, eq=False)
class StorageConstraint:
    id: int
    target: Expr
    storage: StorageKind
    source_loc: SourceLocation
    provenance: ConstraintProvenance = ConstraintProvenance.AUTHOR

    def __eq__(self, other) -> bool:
        if not isinstance(other, StorageConstraint):
            return NotImplemented
        return (
            self.id == other.id
            and self.target is other.target
            and self.storage is other.storage
            and self.source_loc == other.source_loc
            and self.provenance is other.provenance
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.id,
                id(self.target),
                self.storage,
                self.source_loc,
                self.provenance,
            )
        )


@dataclass(frozen=True, slots=True)
class ConstraintList:
    items: tuple[StorageConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


__all__ = [
    "ConstraintList",
    "ConstraintProvenance",
    "SourceLocation",
    "StorageConstraint",
]
