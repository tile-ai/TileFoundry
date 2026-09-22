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
    MemoryMetadata,
    ComputeCostMetadata,
    Footprint,
    MemoryLevelPeak,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    Traffic,
    Breakdown,
    Spread,
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
    "MemoryMetadata",
    "ComputeCostMetadata",
    "ExplicitMemoryLevelFacts",
    "ImplicitMemoryLevelFacts",
    "Footprint",
    "MemoryHierarchyFacts",
    "MemoryLevelPeak",
    "MemoryLevelRelation",
    "RegionMemoryMetadata",
    "MemoryRelationKind",
    "ParallelCapacityFacts",
    "PerformanceServiceFacts",
    "PerformanceMetadata",
    "PerformanceSummaryMetadata",
    "RooflineMetadata",
    "Traffic",
    "Breakdown",
    "Spread",
    "ThroughputFacts",
    "TimelineMetadata",
    "TrafficBytes",
    "ValueLifetime",
    "analyze",
    "check_program",
]
