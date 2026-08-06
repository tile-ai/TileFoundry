"""The immutable service descriptor selected by a Target for Analyze."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from tilefoundry.ir.core import IRMetadata

# An algorithm runs over the selected Module and Function against one target.
AnalysisCallable = Callable[
    ["Module", "Function", "Target", object | None], None
]


@dataclass(frozen=True)
class Analyzer:
    """One analysis service: its identity, dependencies, and owned Metadata."""

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[IRMetadata], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("an analysis needs a non-empty selector")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(
                f"{self.selector}: duplicate entries in requires {list(self.requires)}"
            )
        if self.selector in self.requires:
            raise ValueError(f"{self.selector}: an analysis cannot require itself")
        for produced in self.produces:
            if not isinstance(produced, type) or not issubclass(produced, IRMetadata):
                raise ValueError(
                    f"{self.selector}: produces must name IRMetadata subclasses, "
                    f"got {produced!r}"
                )
        if len(set(self.produces)) != len(self.produces):
            raise ValueError(
                f"{self.selector}: the same Metadata type is produced twice"
            )


@cache
def builtin_analyzers() -> dict[str, Analyzer]:
    """Return the immutable services inherited by built-in Target classes."""
    from tilefoundry.analysis.compute_cost import (  # noqa: PLC0415
        SELECTOR as COMPUTE_COST,
    )
    from tilefoundry.analysis.compute_cost import (  # noqa: PLC0415
        analyze_compute_cost,
    )
    from tilefoundry.analysis.memory import (  # noqa: PLC0415
        SELECTOR as MEMORY,
    )
    from tilefoundry.analysis.memory import (  # noqa: PLC0415
        analyze_memory,
    )
    from tilefoundry.analysis.metadata import (  # noqa: PLC0415
        ComputeCostMetadata,
        MemoryMetadata,
        RooflineMetadata,
        TimelineMetadata,
    )
    from tilefoundry.analysis.roofline import (  # noqa: PLC0415
        SELECTOR as ROOFLINE,
    )
    from tilefoundry.analysis.roofline import (  # noqa: PLC0415
        analyze_roofline,
    )
    from tilefoundry.analysis.timeline import (  # noqa: PLC0415
        SELECTOR as TIMELINE,
    )
    from tilefoundry.analysis.timeline import (  # noqa: PLC0415
        analyze_timeline,
    )

    return {
        COMPUTE_COST: Analyzer(
            COMPUTE_COST,
            analyze_compute_cost,
            produces=(ComputeCostMetadata,),
        ),
        MEMORY: Analyzer(
            MEMORY,
            analyze_memory,
            requires=(COMPUTE_COST,),
            produces=(MemoryMetadata,),
        ),
        ROOFLINE: Analyzer(
            ROOFLINE,
            analyze_roofline,
            requires=(MEMORY, COMPUTE_COST),
            produces=(RooflineMetadata,),
        ),
        TIMELINE: Analyzer(
            TIMELINE,
            analyze_timeline,
            requires=(ROOFLINE,),
            produces=(TimelineMetadata,),
        ),
    }


__all__ = [
    "AnalysisCallable",
    "Analyzer",
    "builtin_analyzers",
]
