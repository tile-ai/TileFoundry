"""Polyhedral graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TileUnit:
    """One statement's identity in a :class:`TileGraph`."""

    name: str
    op: object


@dataclass(frozen=True)
class TileGraph:
    """Represent one HIR function body as a polyhedral analysis result."""

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    buffer_dtypes: dict = field(default_factory=dict)
    parallel_dims: dict = field(default_factory=dict)


__all__ = ["TileGraph", "TileUnit"]
