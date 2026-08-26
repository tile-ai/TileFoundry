"""Polyhedral graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import isl


@dataclass(frozen=True)
class TileUnit:
    """One statement's identity.

    ``name`` is the isl tuple name shared by this statement's pieces of
    ``TileGraph.domain``/``reads``/``writes``/``deps`` (e.g. ``"MM"``).
    ``op`` is the HIR ``Call`` (op@site) that produced this statement --
    the call node itself, not just its bare ``Op``, so a consumer can
    still recover ``op.target`` / ``op.args`` / ``op.type``.
    """

    name: str
    op: object


@dataclass(frozen=True)
class TileGraph:
    """Represent one HIR function body as a polyhedral analysis result.

    Domain and access unions use one tuple name per statement or buffer.
    ``deps`` contains inferred RAW must-dependencies and ``params`` resolves
    dynamic ISL parameters. Buffer dtypes support byte counts without another
    HIR walk; ``parallel_dims`` reports dependence-free statement dimensions.
    Scheduling owns all schedule trees and resource decisions.
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    buffer_dtypes: dict = field(default_factory=dict)
    parallel_dims: dict = field(default_factory=dict)


__all__ = ["TileGraph", "TileUnit"]
