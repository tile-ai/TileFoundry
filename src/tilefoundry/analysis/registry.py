"""The immutable service descriptor selected by a Target for Analyze."""

from __future__ import annotations

from tilefoundry.target.services import AnalysisCallable, Analyzer


def builtin_analyzer(selector: str) -> Analyzer | None:
    """Construct the one standard analysis service named by *selector*."""
    if selector == "compute-cost":
        from tilefoundry.analysis.compute_cost import analyze_compute_cost  # noqa: PLC0415
        from tilefoundry.analysis.metadata import ComputeCostMetadata  # noqa: PLC0415

        return Analyzer(
            "compute-cost", analyze_compute_cost, produces=(ComputeCostMetadata,)
        )
    if selector == "memory":
        from tilefoundry.analysis.memory import analyze_memory  # noqa: PLC0415
        from tilefoundry.analysis.metadata import (  # noqa: PLC0415
            LoopFootprintMetadata,
            MemoryMetadata,
        )

        return Analyzer(
            "memory",
            analyze_memory,
            requires=("compute-cost",),
            produces=(MemoryMetadata, LoopFootprintMetadata),
        )
    if selector == "roofline":
        from tilefoundry.analysis.metadata import RooflineMetadata  # noqa: PLC0415
        from tilefoundry.analysis.roofline import analyze_roofline  # noqa: PLC0415

        return Analyzer(
            "roofline",
            analyze_roofline,
            requires=("memory", "compute-cost"),
            produces=(RooflineMetadata,),
        )
    if selector == "timeline":
        from tilefoundry.analysis.metadata import (  # noqa: PLC0415
            TimelineMetadata,
            TimelineSummaryMetadata,
        )
        from tilefoundry.analysis.timeline import analyze_timeline  # noqa: PLC0415

        return Analyzer(
            "timeline",
            analyze_timeline,
            requires=("compute-cost",),
            produces=(TimelineMetadata, TimelineSummaryMetadata),
        )
    return None


__all__ = [
    "AnalysisCallable",
    "Analyzer",
    "builtin_analyzer",
]
