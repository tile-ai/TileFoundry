"""Direct public Analysis fact contract."""

from __future__ import annotations

# ruff: noqa: I001 -- curated public import order.

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
from .registry import Analyzer
from .api import AnalysisResult, analyze
from .check import check_program
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
    "AccessFootprint",
    "Analyzer",
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
    "check_program",
    "extract",
    "statement_time_dims",
    "time_extents",
]
