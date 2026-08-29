"""Place modeled work by querying the shared lexical Scope tree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.visitor import ExprVisitor

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
from .scope import Scope
from .visitor import AnalyzeContext

SELECTOR = "performance"


@dataclass
class PerformanceContext(AnalyzeContext):
    """State carried through the performance-family expression walk."""

    facts: ThroughputFacts | None = None
    services: PerformanceServiceFacts | None = None
    occurrences: list[tuple[Scope, Call, int]] = field(default_factory=list)


class PerformanceVisitor(ExprVisitor[None]):
    """Collect one duration per Call in authored order."""

    def visit_GridRegionExpr(
        self, expr: GridRegionExpr, ctx: PerformanceContext
    ) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.init_args:
            self.visit(operand, ctx)
        self.visit(expr.body, inner)
        for operand in expr.yield_values:
            self.visit(operand, inner)

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: PerformanceContext
    ) -> None:
        if not isinstance(expr, Call):
            return
        scope = ctx.current if id(expr) in ctx.current.accesses["narrow"] else ctx.root
        cost = get_metadata(expr, ComputeCostMetadata)
        moved = get_metadata(expr, TrafficMetadata)
        if cost is None or moved is None:
            raise AnalysisError(f"performance: missing compute/memory record for {expr!r}")
        if ctx.facts is None or ctx.services is None:
            raise AnalysisError("performance: visitor context is missing target facts")
        duration = _local_duration_ns(
            cost,
            ctx.facts,
            ctx.services,
            moved=moved,
            level=ctx.level,
        )
        ctx.occurrences.append((scope, expr, duration))


def analyze_performance(function: Function, context: AnalyzeContext) -> None:
    """Attach flat occurrence intervals and one function envelope."""
    if context.level is None:
        raise AnalysisError("performance requires a resolved topology level")
    facts = context.target.get_facts(ThroughputFacts)
    services = context.target.get_facts(PerformanceServiceFacts)
    performance_context = PerformanceContext(
        module=context.module,
        target=context.target,
        level=context.level,
        options=context.options,
        root=context.root,
        current=context.current,
        facts=facts,
        services=services,
    )
    PerformanceVisitor().visit(function.body, performance_context)
    scope_periods: dict[object, int] = defaultdict(int)
    for scope, call, duration in performance_context.occurrences:
        if not (
            scope.depth == 2
            and isinstance(call.target, Binary)
            and getattr(call.type, "shape", ()) == ()
        ):
            scope_periods[scope] += duration
    cursor = 0
    total_cursor = 0
    records = []
    for scope, call, duration in performance_context.occurrences:
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
