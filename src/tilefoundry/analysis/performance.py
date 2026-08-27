"""Place modeled work by querying the shared lexical Scope tree."""

from __future__ import annotations

from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.function import Function

from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .metadata import (
    ComputeCostMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    TimelineMetadata,
    TrafficMetadata,
)
from .scope import walk_scopes
from .visitor import AnalyzeContext
from .walk import attach, postorder, reachable_functions

SELECTOR = "performance"


def analyze_performance(function: Function, context: AnalyzeContext) -> None:
    """Attach flat occurrence intervals and one function envelope."""
    if context.level is None:
        raise AnalysisError("performance requires a resolved topology level")
    facts = context.target.get_facts(ThroughputFacts)
    services = context.target.get_facts(PerformanceServiceFacts)
    for fn in reachable_functions(function):
        scopes = tuple(walk_scopes(context.root))
        local = []
        for call in postorder(fn.body):
            if not isinstance(call, Call):
                continue
            scope = next(
                (
                    candidate
                    for candidate in scopes
                    if any(item is call for item in candidate.accesses.get("narrow", {}))
                ),
                context.root,
            )
            cost = get_metadata(call, ComputeCostMetadata)
            moved = get_metadata(call, TrafficMetadata)
            if cost is None or moved is None:
                raise AnalysisError(
                    f"performance: missing compute/memory record for {call!r}"
                )
            duration = _local_duration_ns(
                cost,
                facts,
                services,
                moved=moved,
                level=context.level,
            )
            local.append((scope, call, duration))
        cursor = 0
        for scope, call, duration in local:
            if not duration:
                continue
            trips = 1
            owner = scope
            while owner.parent is not None:
                if owner.is_variant(call):
                    trips *= max(1, owner.trips())
                owner = owner.parent
            scheduled = duration * trips
            attach(
                call,
                PerformanceMetadata(
                    TimelineMetadata(
                        start_ns=cursor,
                        end_ns=cursor + scheduled,
                        trips=1,
                        stride_ns=0,
                    )
                ),
            )
            cursor += scheduled
        attach(
            fn,
            PerformanceSummaryMetadata(
                timeline=TimelineMetadata(start_ns=0, end_ns=cursor),
                waves=1,
            ),
        )


__all__ = ["SELECTOR", "analyze_performance"]
