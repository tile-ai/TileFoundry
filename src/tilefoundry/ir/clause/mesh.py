"""Mesh hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import types

from .base import Where


@dataclass(frozen=True)
class Mesh(Where):
    """Filter an eventual ShardLayout by one existing Mesh value."""

    mesh: types.Mesh | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, types.Mesh):
            raise TypeError(f"mesh constraint requires a Mesh, got {type(self.mesh).__name__}")


__all__ = ["Mesh"]
