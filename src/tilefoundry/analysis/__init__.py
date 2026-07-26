"""Direct public Analysis fact contract."""

from __future__ import annotations

from typing import Protocol

from .analyzer import AnalysisError, AnalysisOptions, AnalysisResult, analyze
from .atom_facts import AtomFact
from .metadata import (
    FootprintMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)
from .poly import (
    AccessFootprint,
    AxisExtent,
    ExtractError,
    TileGraph,
    TileUnit,
    access_footprints,
    carried_distances,
    extract,
    statement_time_dims,
    time_extents,
)


class Analysis(Protocol):
    """One stage's target-dependent facts, for the Schedule stage that
    decides over them: the polyhedral model itself is target-independent
    (:func:`extract`), the atom catalogue and the store a tile lives in are
    not. That store belongs to the level, not the device -- an AMX tile at the
    ``core`` level lives in L1d, a CUDA one at the ``cta`` level in shared
    memory."""

    stage: str
    tile_capacity_bytes: int

    def candidate_atoms(self, op: "Call") -> list[AtomFact]: ...


__all__ = [
    "AccessFootprint",
    "AxisExtent",
    "Analysis",
    "AnalysisError",
    "AnalysisOptions",
    "AnalysisResult",
    "AtomFact",
    "ExtractError",
    "FootprintMetadata",
    "RooflineMetadata",
    "TileGraph",
    "TileUnit",
    "TimelineMetadata",
    "TrafficBytes",
    "access_footprints",
    "analyze",
    "carried_distances",
    "extract",
    "statement_time_dims",
    "time_extents",
]
