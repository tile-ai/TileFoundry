"""Physical-layout hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import types

from .base import Where


@dataclass(frozen=True)
class LayoutWildcard:
    """Private wildcard value used only in constraint-owned Layouts."""

    def __repr__(self) -> str:
        return "_"


LAYOUT_WILDCARD = LayoutWildcard()


def is_layout_wildcard(value: object) -> bool:
    """Return whether ``value`` is the private constraint wildcard."""
    return type(value) is LayoutWildcard


@dataclass(frozen=True)
class Layout(Where):
    """Fix a Layout pattern and its authored ShardAttr bindings."""

    layout: types.Layout = types.Layout(shape=())
    bindings: tuple[tuple[str, types.ShardAttr], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layout, types.Layout):
            raise TypeError(f"layout constraint requires Layout, got {type(self.layout).__name__}")
        bindings = tuple(self.bindings)
        for topology, attr in bindings:
            if not isinstance(topology, str) or not topology:
                raise ValueError("layout binding topology must be non-empty")
            if not isinstance(attr, types.ShardAttr):
                raise TypeError(f"layout binding requires ShardAttr, got {type(attr).__name__}")
        if len({topology for topology, _ in bindings}) != len(bindings):
            raise ValueError("layout constraint cannot bind one topology more than once")
        object.__setattr__(self, "bindings", bindings)

    @property
    def physical_shape(self) -> tuple:
        """Return the constraint-owned physical shape pattern."""
        return self.layout.shape


__all__ = ["LAYOUT_WILDCARD", "Layout", "is_layout_wildcard"]
