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
    PerformanceServiceFacts,
    ThroughputFacts,
)
from .metadata import (
    BufferFootprint,
    ComputeCostMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    AllocationMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TrafficMetadata,
    TimelineMetadata,
    TrafficBytes,
    ValueLifetime,
)
from .registry import Analyzer
from .api import AnalysisResult, analyze
from .check import check_program


__all__ = [
    "Analyzer",
    "AnalysisError",
    "AnalysisResult",
    "BufferFootprint",
    "ComputeCostMetadata",
    "ExplicitMemoryLevelFacts",
    "ImplicitMemoryLevelFacts",
    "LevelFootprint",
    "LoopFootprintMetadata",
    "MemoryHierarchyFacts",
    "MemoryLevelRelation",
    "AllocationMetadata",
    "MemoryMetadata",
    "MemoryRelationKind",
    "ParallelCapacityFacts",
    "PerformanceServiceFacts",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RooflineMetadata",
    "TrafficMetadata",
    "ThroughputFacts",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
    "analyze",
    "check_program",
]
