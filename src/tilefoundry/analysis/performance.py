"""Place modeled work by querying the shared lexical Scope tree."""

from __future__ import annotations

from collections import defaultdict

from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.visitor import collect_exprs

from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import ParallelCapacityFacts, PerformanceServiceFacts, ThroughputFacts
from .metadata import (
    ComputeCostMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficMetadata,
)
from .scope import walk_scopes
from .visitor import AnalyzeContext

SELECTOR = "performance"


def analyze_performance(function: Function, context: AnalyzeContext) -> None:
    """Attach flat occurrence intervals and one function envelope."""
    if context.level is None:
        raise AnalysisError("performance requires a resolved topology level")
    facts = context.target.get_facts(ThroughputFacts)
    services = context.target.get_facts(PerformanceServiceFacts)
    scopes = tuple(walk_scopes(context.root))
    local = []
    for call in collect_exprs(function.body):
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
            raise AnalysisError(f"performance: missing compute/memory record for {call!r}")
        duration = _local_duration_ns(
            cost,
            facts,
            services,
            moved=moved,
            level=context.level,
        )
        local.append((scope, call, duration))
    scope_periods: dict[object, int] = defaultdict(int)
    for scope, call, duration in local:
        if not (
            scope.depth == 2
            and isinstance(call.target, Binary)
            and getattr(call.type, "shape", ()) == ()
        ):
            scope_periods[scope] += duration
    cursor = 0
    total_cursor = 0
    records = []
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
        serialize_boundary = (
            (scope.depth >= 4 and getattr(call.type, "shape", ()) != ())
            or (
                scope.depth == 2
                and isinstance(call.target, Binary)
                and getattr(call.type, "shape", ()) == ()
            )
        )
        start = cursor
        if serialize_boundary:
            trips = 1
            stride = 0
            end = start + scheduled
            cursor += scheduled
        else:
            stride = scope_periods[scope] if trips > 1 else 0
            end = start + duration
            cursor += duration if scope.depth < 4 else scheduled
        records.append((call, start, end, trips, stride, scheduled))
        total_cursor += scheduled
    for call, start, end, trips, stride, scheduled in records:
        if end + (trips - 1) * stride > total_cursor:
            end, trips, stride = start + scheduled, 1, 0
        attach(
            call,
            PerformanceMetadata(
                TimelineMetadata(
                    start_ns=start,
                    end_ns=end,
                    trips=trips,
                    stride_ns=stride,
                )
            ),
        )
    summary_end = total_cursor
    roofline = get_metadata(function, RooflineMetadata)
    if roofline is not None:
        summary_end = max(summary_end, roofline.ideal_ns)
    placement = context.target.get_facts(ParallelCapacityFacts)
    topology = context.module.resolve_topology(placement.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent "
            f"{topology.size!r}"
        )
    waves = -(-topology_extent // placement.parallel_units)
    attach(
        function,
        PerformanceSummaryMetadata(
            timeline=TimelineMetadata(start_ns=0, end_ns=summary_end * waves),
            waves=waves,
        ),
    )


__all__ = ["SELECTOR", "analyze_performance"]
