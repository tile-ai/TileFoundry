"""Physical-layout hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.shard import Layout, ShardAttr

from .base import ScheduleConstraint


@dataclass(frozen=True)
class _LayoutWildcard:
    """Private wildcard value used only in constraint-owned Layouts."""

    def __repr__(self) -> str:
        return "_"


_LAYOUT_WILDCARD = _LayoutWildcard()


def is_layout_wildcard(value: object) -> bool:
    """Return whether ``value`` is the private constraint wildcard."""
    return type(value) is _LayoutWildcard


@dataclass(frozen=True)
class LayoutConstraint(ScheduleConstraint):
    """Fix a Layout pattern and its authored ShardAttr bindings."""

    layout: Layout = Layout(shape=())
    bindings: tuple[tuple[str, ShardAttr], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Layout):
            raise TypeError(
                f"layout constraint requires Layout, got "
                f"{type(self.layout).__name__}"
            )
        bindings = tuple(self.bindings)
        for topology, attr in bindings:
            if not isinstance(topology, str) or not topology:
                raise ValueError("layout binding topology must be non-empty")
            if not isinstance(attr, ShardAttr):
                raise TypeError(
                    f"layout binding requires ShardAttr, got {type(attr).__name__}"
                )
        if len({topology for topology, _ in bindings}) != len(bindings):
            raise ValueError("layout constraint cannot bind one topology more than once")
        object.__setattr__(self, "bindings", bindings)

    @property
    def physical_shape(self) -> tuple:
        """Return the constraint-owned physical shape pattern."""
        return self.layout.shape


__all__ = [
    "LayoutConstraint",
    "is_layout_wildcard",
]
