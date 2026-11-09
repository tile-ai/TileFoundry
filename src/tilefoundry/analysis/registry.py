"""The immutable service descriptor selected by a Target for Analyze."""

from __future__ import annotations

from tilefoundry import services, target

from .allocation import AllocationSummaryMetadata
from .check import PerformanceChecker
from .compute_cost import analyze_compute_cost
from .liveness import LivenessMetadata
from .memory import analyze_memory
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from .performance import analyze_performance
from .roofline import analyze_roofline


class PerformanceAnalyzer(target.Analyzer):
    """The performance service and the program requirements it owns."""

    def get_checker(self) -> target.AnalysisChecker:
        """Return a fresh visitor so one checked program cannot memoize another."""
        return PerformanceChecker()


def builtin_analyzer(selector: str) -> target.Analyzer | None:
    """Construct the one standard analysis service named by *selector*."""
    if selector == "compute-cost":
        return target.Analyzer(
            "compute-cost",
            analyze_compute_cost,
            produces=(ComputeCostMetadata,),
        )
    if selector == "memory":
        return target.Analyzer(
            "memory",
            analyze_memory,
            produces=(
                MemoryMetadata,
                RegionMemoryMetadata,
                LivenessMetadata,
                AllocationSummaryMetadata,
            ),
        )
    if selector == "roofline":
        return target.Analyzer(
            "roofline",
            analyze_roofline,
            requires=("compute-cost", "memory"),
            produces=(RooflineMetadata,),
        )
    if selector == "performance":
        return PerformanceAnalyzer(
            "performance",
            analyze_performance,
            requires=("compute-cost", "memory", "roofline"),
            produces=(PerformanceMetadata, PerformanceSummaryMetadata),
        )
    return None


services.registry.register(services.builtin_analyzer, builtin_analyzer)

__all__ = ["builtin_analyzer"]
