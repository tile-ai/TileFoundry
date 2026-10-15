"""Place modeled work by querying the shared IterationScope tree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.target import UnsupportedCapabilityError
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor, ValuationVisitor

from .compute_cost import local_duration_ns
from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .iteration_scope import IterationScope
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TimelineMetadata,
)
from .visitor import AnalyzeContext

SELECTOR = "performance"


@dataclass
class PerformanceContext(AnalyzeContext):
    """State carried through the performance-family expression walk."""

    facts: ThroughputFacts | None = None
    services: PerformanceServiceFacts | None = None
    occurrences: list[tuple[IterationScope, Call, int]] = field(default_factory=list)


class PerformanceVisitor(ValuationVisitor[None]):
    """Collect one duration per Call in authored order."""

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        callee = CallFrameVisitor.callee_frame(call, ctx.frame)
        if callee is None:
            return self.default_visit_leaf(call, (), ctx)
        duration = ctx.definitions[id(callee.function)].schedule_ns
        if duration is None:
            raise AnalysisError(f"performance: missing summary for {callee.function.name!r}")
        ctx.occurrences.append((ctx.current, call, duration))


    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: PerformanceContext
    ) -> None:
        if not isinstance(expr, Call):
            return
        scope = ctx.current
        cost = get_metadata(expr, ComputeCostMetadata)
        moved = get_metadata(expr, MemoryMetadata)
        if cost is None or moved is None:
            raise AnalysisError(f"performance: missing compute/memory record for {expr!r}")
        if ctx.facts is None or ctx.services is None:
            raise AnalysisError("performance: visitor context is missing target facts")
        duration = local_duration_ns(
            cost,
            ctx.facts,
            ctx.services,
            moved=moved,
            topology_level=ctx.topology_level,
        )
        ctx.occurrences.append((scope, expr, duration))


def schedule(occurrences, context, *, origin):
    """Schedule one definition relative to its function or call scope."""
    scope_periods: dict[object, int] = defaultdict(int)
    for scope, call, duration in occurrences:
        if not (
            (scope.depth - origin.depth) == 2
            and isinstance(call.target, Binary)
            and getattr(call.type, "shape", ()) == ()
        ):
            scope_periods[scope] += duration

    cursor = 0
    total_cursor = 0
    records = []
    for scope, call, duration in occurrences:
        if not duration:
            continue
        trips = scope.repeats_of(call, context.topology_level, stop=origin).loop_trips
        scheduled = duration * trips
        serialize_boundary = (
            (scope.depth - origin.depth) >= 4 and getattr(call.type, "shape", ()) != ()
        ) or (
            (scope.depth - origin.depth) == 2
            and isinstance(call.target, Binary)
            and getattr(call.type, "shape", ()) == ()
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
            cursor += duration if (scope.depth - origin.depth) < 4 else scheduled
        records.append((call, start, end, trips, stride, scheduled))
        total_cursor += scheduled
    for call, start, end, trips, stride, scheduled in records:
        if end + (trips - 1) * stride > total_cursor:
            end, trips, stride = start + scheduled, 1, 0
        context.call_times[id(call)] = start
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
    return total_cursor


def _analyze_performance(function: Function, context: AnalyzeContext) -> None:
    """Attach flat occurrence intervals and one function envelope."""
    if context.topology_level is None:
        raise AnalysisError("performance requires a resolved topology level")
    facts = context.target.get_facts(ThroughputFacts)
    services = context.target.get_facts(PerformanceServiceFacts, context.topology_level)
    performance_context = PerformanceContext(**vars(context), facts=facts, services=services)
    PerformanceVisitor().visit(function.body, performance_context)
    total_cursor = schedule(performance_context.occurrences, context, origin=context.root)
    context.schedule_ns = total_cursor
    summary_end = total_cursor
    roofline = get_metadata(function, RooflineMetadata)
    if context.root.is_root_definition and roofline is not None:
        summary_end = max(summary_end, roofline.ideal_ns)
    level = context.target.get_facts(TopologyFacts).parallel()
    if level is None or level.max_physical_units is None:
        raise UnsupportedCapabilityError(
            f"{type(context.target).__name__}: no default parallel topology level"
        )
    topology = next(item for item in context.topologies if item.name == level.name)
    topology_extent = (
        static_dim_value(topology.size)
        if context.root.is_root_definition
        else max(scope.mesh_positions(level.name) for scope in context.node_scopes.values())
    )
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    waves = -(-topology_extent // level.max_physical_units)
    attach(
        function,
        PerformanceSummaryMetadata(
            timeline=TimelineMetadata(start_ns=0, end_ns=summary_end * waves),
            waves=waves,
        ),
    )


def analyze_performance(function: Function, context: AnalyzeContext) -> None:
    """Schedule each definition once in local coordinates."""
    context.run_definitions(SELECTOR, _analyze_performance)


__all__ = ["SELECTOR", "analyze_performance"]
