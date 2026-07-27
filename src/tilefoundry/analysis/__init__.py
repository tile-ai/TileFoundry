"""Direct public Analysis fact contract."""

from __future__ import annotations

# ruff: noqa: I001 -- curated order: the families register themselves on import
# and must load after the registry they register into.

from typing import Protocol

from .errors import AnalysisError
from .facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    ParallelCapacityFacts,
    ThroughputFacts,
)
from .metadata import (
    ComputeCostMetadata,
    LevelFootprint,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
    ValueLifetime,
)
from .registry import ANALYSES, AnalysisAlgorithm, register_analysis
from . import compute_cost, memory, roofline, timeline  # noqa: F401
from .analyzer import AnalysisOptions, AnalysisResult, analyze
from .atom_facts import AtomFact
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
    "ANALYSES",
    "AccessFootprint",
    "Analysis",
    "AnalysisAlgorithm",
    "AnalysisError",
    "AnalysisOptions",
    "AnalysisResult",
    "AtomFact",
    "AxisExtent",
    "ComputeCostMetadata",
    "ExplicitMemoryLevelFacts",
    "ExtractError",
    "ImplicitMemoryLevelFacts",
    "LevelFootprint",
    "MemoryHierarchyFacts",
    "MemoryLevelRelation",
    "MemoryMetadata",
    "MemoryRelationKind",
    "ParallelCapacityFacts",
    "RooflineMetadata",
    "ThroughputFacts",
    "TileGraph",
    "TileUnit",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
    "access_footprints",
    "analyze",
    "carried_distances",
    "extract",
    "register_analysis",
    "statement_time_dims",
    "time_extents",
]
