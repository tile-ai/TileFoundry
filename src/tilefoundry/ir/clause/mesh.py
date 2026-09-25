"""Mesh hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import Mesh

from .base import WhereClause


@dataclass(frozen=True)
class MeshClause(WhereClause):
    """Filter an eventual ShardLayout by one existing Mesh value."""

    mesh: Mesh | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, Mesh):
            raise TypeError(f"mesh constraint requires a Mesh, got {type(self.mesh).__name__}")


__all__ = ["MeshClause"]
