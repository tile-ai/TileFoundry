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
    BufferFootprint,
    ComputeCostMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    MemoryMetadata,
    OccurrenceProvenance,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
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
    "BufferFootprint",
    "ComputeCostMetadata",
    "ExplicitMemoryLevelFacts",
    "ExtractError",
    "ImplicitMemoryLevelFacts",
    "LevelFootprint",
    "LoopFootprintMetadata",
    "MemoryHierarchyFacts",
    "MemoryLevelRelation",
    "MemoryMetadata",
    "MemoryRelationKind",
    "OccurrenceProvenance",
    "ParallelCapacityFacts",
    "RooflineMetadata",
    "ThroughputFacts",
    "TileGraph",
    "TileUnit",
    "TimelineMetadata",
    "TimelineSummaryMetadata",
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
