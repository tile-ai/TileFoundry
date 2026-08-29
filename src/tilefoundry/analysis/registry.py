"""The immutable service descriptor selected by a Target for Analyze."""

from __future__ import annotations

from tilefoundry.target.services import AnalysisCallable, AnalysisChecker, Analyzer


class _PerformanceAnalyzer(Analyzer):
    """The performance service and the program requirements it owns."""

    def get_checker(self) -> AnalysisChecker:
        """Return a fresh visitor so one checked program cannot memoize another."""
        from tilefoundry.analysis.check import PerformanceChecker  # noqa: PLC0415

        return PerformanceChecker()


def builtin_analyzer(selector: str) -> Analyzer | None:
    """Construct the one standard analysis service named by *selector*."""
    if selector == "compute-cost":
        from tilefoundry.analysis.compute_cost import analyze_compute_cost  # noqa: PLC0415
        from tilefoundry.analysis.metadata import ComputeCostMetadata  # noqa: PLC0415

        return Analyzer(
            "compute-cost",
            analyze_compute_cost,
            produces=(ComputeCostMetadata,),
        )
    if selector == "memory":
        from tilefoundry.analysis.memory import analyze_memory  # noqa: PLC0415
        from tilefoundry.analysis.metadata import (  # noqa: PLC0415
            LoopFootprintMetadata,
            MemoryMetadata,
            TrafficMetadata,
        )

        return Analyzer(
            "memory",
            analyze_memory,
            produces=(MemoryMetadata, LoopFootprintMetadata, TrafficMetadata),
        )
    if selector == "roofline":
        from tilefoundry.analysis.metadata import RooflineMetadata  # noqa: PLC0415
        from tilefoundry.analysis.roofline import analyze_roofline  # noqa: PLC0415

        return Analyzer(
            "roofline",
            analyze_roofline,
            requires=("compute-cost", "memory"),
            produces=(RooflineMetadata,),
        )
    if selector == "performance":
        from tilefoundry.analysis.metadata import (  # noqa: PLC0415
            PerformanceMetadata,
            PerformanceSummaryMetadata,
        )
        from tilefoundry.analysis.performance import analyze_performance  # noqa: PLC0415

        return _PerformanceAnalyzer(
            "performance",
            analyze_performance,
            requires=("compute-cost", "memory"),
            produces=(PerformanceMetadata, PerformanceSummaryMetadata),
        )
    return None


__all__ = [
    "AnalysisCallable",
    "Analyzer",
    "builtin_analyzer",
]
