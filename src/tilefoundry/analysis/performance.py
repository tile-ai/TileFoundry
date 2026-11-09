"""Place modeled work by querying the shared IterationScope tree."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tilefoundry import target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import math

from .compute_cost import inflight_bytes, local_duration_ns
from .errors import Error
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

    facts: target.ThroughputFacts | None = None
    active: int = 1
    parallel: target.TopologyLevelFacts | None = None
    occurrences: list[tuple[IterationScope, expr.Call, int]] = field(default_factory=list)


class PerformanceVisitor(valuation.Visitor[None]):
    """Collect one duration per Call in authored order."""

    def visit_Call(self, call, ctx):
        self.visit_operands(call, ctx)
        callee = ctx.callee
        if callee is None:
            return self.default_visit_leaf(call, (), ctx)
        summary = expr.get_metadata(callee, PerformanceSummaryMetadata, ctx=ctx)
        if summary is None:
            raise Error(f"performance: missing summary for {callee.name!r}")
        waves = wave_count(ctx.definitions[id(callee)], ctx.parallel)
        duration = (summary.timeline.end_ns - summary.timeline.start_ns) // waves
        ctx.occurrences.append((ctx.scope, call, duration))

    def default_visit_leaf(
        self, node: expr.Expr, operands: tuple[None, ...], ctx: PerformanceContext
    ) -> None:
        if not isinstance(node, expr.Call):
            return
        scope = ctx.scope
        cost = expr.get_metadata(node, ComputeCostMetadata)
        moved = expr.get_metadata(node, MemoryMetadata)
        if cost is None or moved is None:
            raise Error(f"performance: missing compute/memory record for {node!r}")
        if ctx.facts is None:
            raise Error("performance: visitor context is missing target facts")
        inflight = inflight_bytes(node, scope, ctx.target, ctx.facts.bandwidth_level)
        duration = local_duration_ns(
            cost,
            ctx.facts,
            ctx.target.get_facts(target.PerformanceServiceFacts, (ctx.topology_level, ctx.active, inflight)),
            moved=moved,
            topology_level=ctx.topology_level,
            positions=min(scope.mesh_positions(ctx.topology_level), ctx.active),
        )
        ctx.occurrences.append((scope, node, duration))


def wave_count(scope, level) -> int:
    """Whole waves of *level*'s physical units the positions *scope* runs need."""
    return -(-scope.active_positions(level.name) // level.max_physical_units)


def schedule(occurrences, context, *, origin):
    """Schedule one definition relative to its function or call scope."""
    scope_periods: dict[object, int] = defaultdict(int)
    for scope, call, duration in occurrences:
        if not (
            (len(scope.loop_scopes) - len(origin.loop_scopes)) == 2
            and isinstance(call.target, math.Binary)
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
            (len(scope.loop_scopes) - len(origin.loop_scopes)) >= 4
            and getattr(call.type, "shape", ()) != ()
        ) or (
            (len(scope.loop_scopes) - len(origin.loop_scopes)) == 2
            and isinstance(call.target, math.Binary)
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
            cursor += (
                duration if (len(scope.loop_scopes) - len(origin.loop_scopes)) < 4 else scheduled
            )
        records.append((call, start, end, trips, stride, scheduled))
        total_cursor += scheduled
    for call, start, end, trips, stride, scheduled in records:
        if end + (trips - 1) * stride > total_cursor:
            end, trips, stride = start + scheduled, 1, 0
        call.attach_metadata(PerformanceMetadata(
                TimelineMetadata(
                    start_ns=start,
                    end_ns=end,
                    trips=trips,
                    stride_ns=stride,
                )
            ))
    return total_cursor


def analyze_performance(function: core.Function, context: AnalyzeContext) -> None:
    """Attach flat occurrence intervals and one function envelope.

    Rates are those of the units the function's meshes run, and the envelope
    repeats once per wave of those positions.
    """
    if context.topology_level is None:
        raise Error("performance requires a resolved topology level")
    topologies = context.target.get_facts(target.TopologyFacts)
    level = topologies.parallel()
    if level is None or level.max_physical_units is None:
        raise target.UnsupportedCapabilityError(
            f"{type(context.target).__name__}: no default parallel topology level"
        )
    positions = context.scope.active_positions(context.topology_level)
    physical = topologies.level(context.topology_level).max_physical_units
    active = min(positions, physical)
    performance_context = PerformanceContext(
        **vars(context),
        facts=context.target.get_facts(target.ThroughputFacts),
        active=active,
        parallel=level,
    )
    PerformanceVisitor().visit(function.body, performance_context)
    total_cursor = schedule(performance_context.occurrences, context, origin=context.scope.root)
    summary_end = total_cursor
    roofline = expr.get_metadata(function, RooflineMetadata)
    if function is context.entry and roofline is not None:
        summary_end = max(summary_end, roofline.ideal_ns)
    function.attach_metadata(PerformanceSummaryMetadata(
            timeline=TimelineMetadata(
                start_ns=0, end_ns=summary_end * wave_count(context.scope, level)
            ),
            wave=(positions, physical),
        ))


__all__ = ["SELECTOR", "analyze_performance"]
