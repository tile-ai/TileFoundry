"""Place modeled work on exact logical participant sets.

Each primitive occurrence occupies every logical position named by its result
placement for one CTA-local duration. Dependencies and intersecting participant
sets constrain the intervals; physical wave scaling is a separate concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.target import Target

from .check import Placement, _timeline_placements
from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import ParallelCapacityFacts, ThroughputFacts
from .metadata import ComputeCostMetadata, TimelineMetadata, TimelineSummaryMetadata
from .walk import (
    attach,
    children,
    describe,
    loop_scopes,
    loop_trip_count,
    postorder,
    reachable_functions,
)

SELECTOR = "timeline"

_SOLVE_SECONDS = 5.0


@dataclass
class _Occurrence:
    """One primitive occurrence or one structured loop in a local schedule."""

    expr: Call | GridRegionExpr
    placement: Placement
    duration_ns: int
    source_index: int
    predecessors: set[int] = field(default_factory=set)
    body: _Schedule | None = None
    trips: int = 1
    start_ns: int = 0
    end_ns: int = 0


@dataclass
class _Schedule:
    """The direct occurrences in one Function or loop body."""

    occurrences: dict[int, _Occurrence]
    external_predecessors: set[int]
    makespan_ns: int = 0

    @property
    def placement(self) -> Placement:
        return frozenset(
            position
            for occurrence in self.occurrences.values()
            for position in occurrence.placement
        )


def _durations(
    fn: Function,
    facts: ThroughputFacts,
    level: str,
) -> dict[int, int]:
    """Return one CTA-local duration for every primitive occurrence."""
    result: dict[int, int] = {}
    for expr in postorder(fn.body):
        if not isinstance(expr, Call):
            continue
        cost = get_metadata(expr, ComputeCostMetadata)
        if cost is None:
            raise AnalysisError(
                f"{describe(expr)}: the timeline needs the compute-cost record this "
                "call was never given"
            )
        result[id(expr)] = _local_duration_ns(cost, facts, level=level)
    return result


def _producer_ids(expr: Expr, schedulable: set[int]) -> set[int]:
    """Find the nearest schedulable producers beneath a value expression."""
    if id(expr) in schedulable:
        return {id(expr)}
    return {producer for child in children(expr) for producer in _producer_ids(child, schedulable)}


def _solve(occurrences: dict[int, _Occurrence]) -> int:
    """Solve one loop-free local scope and return its makespan."""
    if not occurrences:
        return 0

    ordered = sorted(occurrences.values(), key=lambda item: item.source_index)
    horizon = sum(item.duration_ns for item in ordered)
    model = cp_model.CpModel()
    starts: dict[int, cp_model.IntVar] = {}
    ends: dict[int, cp_model.IntVar] = {}
    intervals_by_position: dict[int, list[cp_model.IntervalVar]] = {}
    for index, occurrence in enumerate(ordered):
        key = id(occurrence.expr)
        start = model.NewIntVar(0, horizon, f"c{index}_start")
        end = model.NewIntVar(0, horizon, f"c{index}_end")
        model.Add(end == start + occurrence.duration_ns)
        starts[key] = start
        ends[key] = end
        if occurrence.duration_ns > 0:
            interval = model.NewIntervalVar(start, occurrence.duration_ns, end, f"c{index}")
            for position in occurrence.placement:
                intervals_by_position.setdefault(position, []).append(interval)

    for occurrence in ordered:
        key = id(occurrence.expr)
        for predecessor in occurrence.predecessors:
            model.Add(starts[key] >= ends[predecessor])
        if occurrence.duration_ns == 0:
            if occurrence.predecessors:
                model.AddMaxEquality(
                    starts[key],
                    [ends[predecessor] for predecessor in occurrence.predecessors],
                )
            else:
                model.Add(starts[key] == 0)

    for intervals in intervals_by_position.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [ends[id(item.expr)] for item in ordered])
    model.Minimize(makespan)
    model.AddDecisionStrategy(
        [starts[id(item.expr)] for item in ordered],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MIN_VALUE,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.max_time_in_seconds = _SOLVE_SECONDS
    if solver.Solve(model) != cp_model.OPTIMAL:
        raise AnalysisError("the participant-set timeline has no optimal schedule")

    optimum = solver.Value(makespan)

    for occurrence in ordered:
        key = id(occurrence.expr)
        occurrence.start_ns = solver.Value(starts[key])
        occurrence.end_ns = solver.Value(ends[key])
    return optimum


def _schedule(
    fn: Function,
    durations: dict[int, int],
    placements: dict[int, Placement],
) -> _Schedule:
    """Build and solve the Function's nested occurrence schedules."""
    values = postorder(fn.body)
    source_index = {id(expr): index for index, expr in enumerate(values)}
    parent, scope_of = loop_scopes(fn)

    schedulable = set(scope_of)

    def representative(producer: int, scope: int | None) -> int:
        producer_scope = scope_of[producer]
        if producer_scope == scope:
            return producer
        child = producer_scope
        while child is not None and parent[child] != scope:
            child = parent[child]
        return producer if child is None else child

    def build(scope: int | None) -> _Schedule:
        direct = [
            expr
            for expr in values
            if isinstance(expr, (Call, GridRegionExpr)) and scope_of[id(expr)] == scope
        ]
        occurrences: dict[int, _Occurrence] = {}
        child_external: dict[int, set[int]] = {}
        for expr in direct:
            if isinstance(expr, Call):
                occurrence = _Occurrence(
                    expr,
                    placements[id(expr)],
                    durations[id(expr)],
                    source_index[id(expr)],
                )
            else:
                body = build(id(expr))
                child_external[id(expr)] = body.external_predecessors
                occurrence = _Occurrence(
                    expr,
                    body.placement,
                    loop_trip_count(expr) * body.makespan_ns,
                    source_index[id(expr)],
                    body=body,
                    trips=loop_trip_count(expr),
                )
            occurrences[id(expr)] = occurrence

        external: set[int] = set()
        for occurrence in occurrences.values():
            expr = occurrence.expr
            operands = expr.args if isinstance(expr, Call) else expr.init_args
            producers = {
                producer for operand in operands for producer in _producer_ids(operand, schedulable)
            }
            if isinstance(expr, GridRegionExpr):
                producers.update(child_external[id(expr)])
            for producer in producers:
                resolved = representative(producer, scope)
                if scope_of[resolved] == scope:
                    occurrence.predecessors.add(resolved)
                else:
                    external.add(resolved)

        plan = _Schedule(occurrences, external)
        plan.makespan_ns = _solve(occurrences)
        return plan

    return build(None)


def _records(
    schedule: _Schedule,
    *,
    offset_ns: int = 0,
    trips: int = 1,
    stride_ns: int = 0,
) -> dict[int, TimelineMetadata]:
    """Materialize absolute first-trip intervals from a nested schedule."""
    result: dict[int, TimelineMetadata] = {}
    for key, occurrence in schedule.occurrences.items():
        start = offset_ns + occurrence.start_ns
        end = offset_ns + occurrence.end_ns
        if isinstance(occurrence.expr, Call):
            result[key] = TimelineMetadata(
                start_ns=start,
                end_ns=end,
                trips=trips,
                stride_ns=stride_ns if trips > 1 else 0,
            )
            continue
        if occurrence.body is not None:
            result.update(
                _records(
                    occurrence.body,
                    offset_ns=start,
                    trips=occurrence.trips,
                    stride_ns=occurrence.body.makespan_ns,
                )
            )
    return result


def analyze_timeline(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Place every reachable Function's occurrences on a local timeline."""
    placement_facts = target.get_facts(ParallelCapacityFacts)
    throughput = target.get_facts(ThroughputFacts)
    topology = module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"timeline: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    waves = -(-topology_extent // placement_facts.parallel_units)
    for fn in reachable_functions(function):
        placements = _timeline_placements(
            module,
            fn,
            placement_facts.topology,
            throughput,
        )
        durations = _durations(fn, throughput, placement_facts.topology)
        schedule = _schedule(fn, durations, placements)
        records = _records(schedule)
        for expr in postorder(fn.body):
            record = records.get(id(expr))
            if record is not None:
                attach(expr, record)
        attach(
            fn,
            TimelineSummaryMetadata(
                local_makespan_ns=schedule.makespan_ns,
                waves=waves,
                estimated_kernel_ns=schedule.makespan_ns * waves,
            ),
        )


__all__ = ["SELECTOR", "analyze_timeline"]
