"""Direct public Analysis fact contract."""

from __future__ import annotations

# ruff: noqa: I001 -- curated order: the families register themselves on import
# and must load after the registry they register into.

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
from .api import AnalysisResult, analyze
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


__all__ = [
    "ANALYSES",
    "AccessFootprint",
    "AnalysisAlgorithm",
    "AnalysisError",
    "AnalysisResult",
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
