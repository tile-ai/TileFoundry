from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base import AgentConstraint


class LayoutDimKind(Enum):
    UNCONSTRAINED = "unconstrained"
    BROADCAST = "broadcast"
    SPLIT = "split"


@dataclass(frozen=True)
class LayoutDimConstraint:
    index: int
    kind: LayoutDimKind
    extent: Any = None
    topology: str | None = None


@dataclass(frozen=True)
class LayoutConstraint(AgentConstraint):
    dims: tuple[LayoutDimConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))


@dataclass(frozen=True)
class PartialConstraint(AgentConstraint):
    reduction: str = "sum"
    topology: str | None = None


__all__ = [
    "LayoutConstraint",
    "LayoutDimConstraint",
    "LayoutDimKind",
    "PartialConstraint",
]
