"""Physical-layout hard constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tilefoundry.ir.types.shard import Broadcast, Partial, ShardAttr, Split

from .base import ScheduleConstraint


class LayoutDimKind(Enum):
    """Surface kind of one physical layout pattern position."""

    EXACT = "exact"
    WILDCARD = "wildcard"
    UNCONSTRAINED = "wildcard"
    BROADCAST = "broadcast"
    SPLIT = "split"


@dataclass(frozen=True)
class LayoutWildcard:
    """Private wildcard sentinel used only by schedule constraints."""

    def __repr__(self) -> str:
        return "_"


WILDCARD = LayoutWildcard()


@dataclass(frozen=True)
class LayoutDimConstraint:
    """One physical layout position and its optional topology binding."""

    index: int
    kind: LayoutDimKind
    extent: Any = None
    topology: str | None = None

    @property
    def value(self) -> Any:
        """Return the exact or symbolic physical extent, if any."""
        return self.extent


@dataclass(frozen=True)
class LayoutConstraint(ScheduleConstraint):
    """Fix a physical-shape pattern and its authored shard attributes."""

    dims: tuple[LayoutDimConstraint, ...] = ()
    attr: ShardAttr | None = None
    attrs: tuple[ShardAttr, ...] = ()
    physical_shape: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        dims = tuple(self.dims)
        attrs = tuple(self.attrs)
        if self.attr is not None and not attrs:
            attrs = (self.attr,)
        if not attrs:
            attrs = tuple(
                Split(dim.index)
                for dim in dims
                if dim.kind is LayoutDimKind.SPLIT
            )
            if not attrs and any(dim.kind is LayoutDimKind.BROADCAST for dim in dims):
                attrs = (Broadcast(),)
        attr = self.attr if self.attr is not None else (attrs[0] if attrs else None)
        physical_shape = tuple(self.physical_shape)
        if not physical_shape:
            physical_shape = tuple(
                WILDCARD if dim.kind in (LayoutDimKind.WILDCARD, LayoutDimKind.UNCONSTRAINED)
                else dim.extent
                for dim in dims
            )
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "attr", attr)
        object.__setattr__(self, "physical_shape", physical_shape)


@dataclass(frozen=True)
class PartialConstraint(ScheduleConstraint):
    """Require a Partial value state with a named reduction."""

    reduction: str = "sum"
    topology: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reduction, str) or not self.reduction:
            raise ValueError("partial reduction must be a non-empty string")

    @property
    def attr(self) -> Partial:
        """Return the corresponding existing IR shard attribute."""
        return Partial(self.reduction)


__all__ = [
    "LayoutConstraint",
    "LayoutDimConstraint",
    "LayoutDimKind",
    "LayoutWildcard",
    "PartialConstraint",
    "WILDCARD",
]
