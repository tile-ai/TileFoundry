"""Place modeled work on exact logical participant sets, in authored order.

Each occurrence holds every position the Mesh it was authored inside names, for
one CTA-local duration. A position runs one occurrence at a time and an
occurrence waits for what it reads, so the program's own placement says what
overlaps. Nothing is searched. Where the buffers sit belongs to ``memory``.

The structural memo owns lexical scopes. ``OccurrencePlan`` is the performance
projection that says where those occurrences materialize, and the flat timeline
refers back to those same scopes rather than copying a second scope tree.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.shape_helpers import static_dim_value

from .check import Placement, _call_placements
from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import ParallelCapacityFacts, PerformanceServiceFacts, ThroughputFacts
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    TimelineMetadata,
    TrafficMetadata,
)
from .visitor import AnalyzeContext, ScopeMemo, StructuralMemo
from .walk import attach, describe, loop_trip_count, reachable_functions

SELECTOR = "performance"


@dataclass(frozen=True)
class OccurrencePlan:
    """One performance occurrence projected from the shared structural memo."""

    expr: Call | GridRegionExpr
    lexical_scope: ScopeMemo
    materialized_scope: ScopeMemo
    placement: Placement
    predecessors: tuple[Call | GridRegionExpr, ...]
    replay_count: int
    duration_ns: int


@dataclass(frozen=True)
class TimelineEntry:
    """One flat first-trip interval within an existing structural scope."""

    expr: Call | GridRegionExpr
    scope: ScopeMemo
    duration_ns: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class ScopeTiming:
    """The local makespan aggregated for one existing structural scope."""

    scope: ScopeMemo
    makespan_ns: int


@dataclass(frozen=True)
class _PerformancePlan:
    function: Function
    occurrences: tuple[OccurrencePlan, ...]


@dataclass(frozen=True)
class _PerformanceTimeline:
    entries: tuple[TimelineEntry, ...]
    scopes: tuple[ScopeTiming, ...]
    waves: int


def _durations(
    fn: Function,
    facts: ThroughputFacts,
    services: PerformanceServiceFacts,
    level: str,
    structural_memo: StructuralMemo,
) -> dict[int, int]:
    """Return one CTA-local duration for every primitive occurrence."""
    result: dict[int, int] = {}
    for expr in structural_memo.definition_order(fn):
        if not isinstance(expr, Call):
            continue
        cost = get_metadata(expr, ComputeCostMetadata)
        if cost is None:
            raise AnalysisError(
                f"{describe(expr)}: performance needs the compute-cost record this "
                "call was never given"
            )
        moved = get_metadata(expr, TrafficMetadata)
        if moved is None:
            raise AnalysisError(
                f"{describe(expr)}: performance needs the traffic record the "
                "memory family states for every call it measures"
            )
        result[id(expr)] = _local_duration_ns(
            cost, facts, services, moved=moved, level=level
        )
    return result


def _producer_occurrences(
    expr: Expr,
    schedulable: dict[int, Call | GridRegionExpr],
    structural_memo: StructuralMemo,
) -> tuple[Call | GridRegionExpr, ...]:
    """Find the nearest schedulable producers beneath a value expression."""
    direct = schedulable.get(id(expr))
    if direct is expr:
        return (direct,)
    found: dict[int, Call | GridRegionExpr] = {}
    for operand in structural_memo.producers(expr):
        for producer in _producer_occurrences(operand, schedulable, structural_memo):
            found[id(producer)] = producer
    return tuple(found.values())


def _variance_paths(
    fn: Function, structural_memo: StructuralMemo
) -> dict[int, tuple[ScopeMemo, ...]]:
    """Return the materialization path each schedulable value varies along."""
    values = structural_memo.definition_order(fn)
    loop_scopes = tuple(
        structural_memo.scope(expr)
        for expr in values
        if isinstance(expr, GridRegionExpr)
    )
    paths: dict[int, tuple[ScopeMemo, ...]] = {}
    for expr in values:
        if not isinstance(expr, (Call, GridRegionExpr)):
            continue
        owners = (scope for scope in loop_scopes if scope.is_variant(expr))
        paths[id(expr)] = tuple(
            sorted(
                owners,
                key=lambda scope: -structural_memo.node(scope.owner).definition_index,
            )
        )
    return paths


def build_performance_plan(
    fn: Function,
    structural_memo: StructuralMemo,
    context: AnalyzeContext,
) -> _PerformancePlan:
    """Project placement, replay, and dependencies onto structural scopes.

    Performance nesting can differ from lexical nesting when invariant work is
    lifted. Materialization paths exist only while assigning the plan's scopes.
    """
    target = context.target
    placement_facts = target.get_facts(ParallelCapacityFacts)
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)
    values = structural_memo.definition_order(fn)
    root_scope = structural_memo.scope(fn)
    paths = _variance_paths(fn, structural_memo)
    placements = _call_placements(context.module, fn, placement_facts.topology)
    durations = _durations(
        fn, throughput, services, placement_facts.topology, structural_memo
    )

    nodes: dict[tuple[ScopeMemo, ...], ScopeMemo] = {(): root_scope}
    hosts: dict[int, ScopeMemo] = {}
    replays: dict[int, int] = {}

    def settle(expr: Call | GridRegionExpr) -> tuple[ScopeMemo, ...]:
        path = paths[id(expr)]
        host_path: tuple[ScopeMemo, ...] = ()
        for size in range(1, len(path) + 1):
            candidate = path[:size]
            if candidate in nodes:
                host_path = candidate
        hosts[id(expr)] = nodes[host_path]
        replays[id(expr)] = 1
        for scope in path[len(host_path) :]:
            if scope.trip_count is not None:
                replays[id(expr)] *= scope.trip_count.value or 1
        return host_path

    loops = tuple(expr for expr in values if isinstance(expr, GridRegionExpr))
    for loop in sorted(
        loops, key=lambda expr: -structural_memo.node(expr).definition_index
    ):
        host_path = settle(loop)
        body_scope = structural_memo.scope(loop)
        nodes[(*host_path, body_scope)] = body_scope
    for expr in values:
        if isinstance(expr, Call):
            settle(expr)

    schedulable = {
        id(expr): expr
        for expr in values
        if isinstance(expr, (Call, GridRegionExpr))
    }
    direct: dict[ScopeMemo, list[Call | GridRegionExpr]] = {}
    for expr in schedulable.values():
        direct.setdefault(hosts[id(expr)], []).append(expr)

    body_scope_by_loop = {id(loop): structural_memo.scope(loop) for loop in loops}
    loop_by_body_scope = {body_scope_by_loop[id(loop)]: loop for loop in loops}

    def is_timeline_predecessor(
        producer: Call | GridRegionExpr, consumer_scope: ScopeMemo
    ) -> bool:
        """Whether this def-use edge completes within the consumer's scope."""
        scope = hosts[id(producer)]
        visited: set[ScopeMemo] = set()
        while scope is not consumer_scope:
            if scope in visited:
                return False
            visited.add(scope)
            owner = loop_by_body_scope.get(scope)
            if owner is None:
                return False
            scope = hosts[id(owner)]
        return True

    def held_under(scope: ScopeMemo) -> Placement:
        found: set[int] = set()
        for expr in direct.get(scope, ()):
            if isinstance(expr, Call):
                found |= placements[id(expr)]
            else:
                found |= held_under(body_scope_by_loop[id(expr)])
        return frozenset(found)

    plans: list[OccurrencePlan] = []
    for expr in schedulable.values():
        operands = expr.args if isinstance(expr, Call) else expr.init_args
        predecessors: dict[int, Call | GridRegionExpr] = {}
        for operand in operands:
            for producer in _producer_occurrences(
                operand, schedulable, structural_memo
            ):
                if producer is not expr and is_timeline_predecessor(
                    producer, hosts[id(expr)]
                ):
                    predecessors[id(producer)] = producer
        materialized_scope = hosts[id(expr)]
        placement = (
            placements[id(expr)]
            if isinstance(expr, Call)
            else held_under(body_scope_by_loop[id(expr)])
        )
        plans.append(
            OccurrencePlan(
                expr=expr,
                lexical_scope=structural_memo.scope_of(expr),
                materialized_scope=materialized_scope,
                placement=placement,
                predecessors=tuple(
                    sorted(
                        predecessors.values(),
                        key=lambda producer: structural_memo.node(
                            producer
                        ).definition_index,
                    )
                ),
                replay_count=replays[id(expr)],
                duration_ns=durations[id(expr)] if isinstance(expr, Call) else 0,
            )
        )
    return _PerformancePlan(fn, tuple(plans))


def build_timeline(
    plan: _PerformancePlan,
    context: AnalyzeContext,
) -> _PerformanceTimeline:
    """Solve a flat timeline whose scopes are the shared structural scopes."""
    structural_memo = context.structural_memo
    plan_by_id = {id(item.expr): item for item in plan.occurrences}
    direct: dict[ScopeMemo, list[OccurrencePlan]] = {}
    for item in plan.occurrences:
        direct.setdefault(item.materialized_scope, []).append(item)
    for items in direct.values():
        items.sort(key=lambda item: structural_memo.node(item.expr).definition_index)

    loop_by_body_scope = {
        structural_memo.scope(item.expr): item
        for item in plan.occurrences
        if isinstance(item.expr, GridRegionExpr)
    }

    def dependency_entry(
        producer: OccurrencePlan, consumer: OccurrencePlan
    ) -> Call | GridRegionExpr:
        """Return the producer completion visible to the consumer's scope."""
        scope = producer.materialized_scope
        visible: Call | GridRegionExpr = producer.expr
        visited: set[ScopeMemo] = set()
        while scope is not consumer.materialized_scope:
            if scope in visited:
                raise AnalysisError(
                    f"{describe(consumer.expr)}: performance plan invariant: "
                    f"scope cycle while resolving predecessor {describe(producer.expr)} "
                    f"from {describe(consumer.materialized_scope.owner)}"
                )
            visited.add(scope)
            owner = loop_by_body_scope.get(scope)
            if owner is None:
                raise AnalysisError(
                    f"{describe(consumer.expr)}: performance plan invariant: "
                    f"predecessor {describe(producer.expr)} materialized in "
                    f"{describe(producer.materialized_scope.owner)} is not visible "
                    f"from {describe(consumer.materialized_scope.owner)}"
                )
            visible = owner.expr
            scope = owner.materialized_scope
        return visible

    resolved_predecessors: dict[
        int, tuple[tuple[OccurrencePlan, Call | GridRegionExpr], ...]
    ] = {}

    def predecessors(
        occurrence: OccurrencePlan,
    ) -> tuple[tuple[OccurrencePlan, Call | GridRegionExpr], ...]:
        """Resolve and validate one occurrence's visible predecessors once."""
        cached = resolved_predecessors.get(id(occurrence.expr))
        if cached is not None:
            return cached
        resolved: list[tuple[OccurrencePlan, Call | GridRegionExpr]] = []
        for predecessor_expr in occurrence.predecessors:
            predecessor = plan_by_id.get(id(predecessor_expr))
            if predecessor is None or predecessor.expr is not predecessor_expr:
                raise AnalysisError(
                    f"{describe(occurrence.expr)}: performance plan invariant: "
                    f"predecessor {describe(predecessor_expr)} has no occurrence plan"
                )
            resolved.append((predecessor, dependency_entry(predecessor, occurrence)))
        result = tuple(resolved)
        resolved_predecessors[id(occurrence.expr)] = result
        return result

    def ordered_occurrences(scope: ScopeMemo) -> tuple[OccurrencePlan, ...]:
        """Stable topological order, preferring definition order among ready work."""
        pending = tuple(direct.get(scope, ()))
        by_id = {id(occurrence.expr): occurrence for occurrence in pending}
        position = {id(occurrence.expr): index for index, occurrence in enumerate(pending)}
        required: dict[int, tuple[Call | GridRegionExpr, ...]] = {}
        dependents: dict[int, list[OccurrencePlan]] = {}
        indegree: dict[int, int] = {}
        for occurrence in pending:
            visible: dict[int, Call | GridRegionExpr] = {}
            for _predecessor, dependency in predecessors(occurrence):
                if dependency is not occurrence.expr:
                    visible[id(dependency)] = dependency
            dependencies = tuple(visible.values())
            key = id(occurrence.expr)
            required[key] = dependencies
            indegree[key] = len(dependencies)
            for dependency in dependencies:
                dependents.setdefault(id(dependency), []).append(occurrence)

        ordered: list[OccurrencePlan] = []
        ready = [
            (position[key], key)
            for key, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        while ready:
            _index, key = heapq.heappop(ready)
            occurrence = by_id[key]
            ordered.append(occurrence)
            for dependent in dependents.get(key, ()):
                dependent_key = id(dependent.expr)
                indegree[dependent_key] -= 1
                if indegree[dependent_key] == 0:
                    heapq.heappush(
                        ready, (position[dependent_key], dependent_key)
                    )
        if len(ordered) != len(pending):
            completed = {id(occurrence.expr) for occurrence in ordered}
            blocked = next(
                occurrence for occurrence in pending if id(occurrence.expr) not in completed
            )
            missing = next(
                dependency
                for dependency in required[id(blocked.expr)]
                if id(dependency) not in completed
            )
            raise AnalysisError(
                f"{describe(blocked.expr)}: performance plan invariant: "
                f"scope {describe(scope.owner)} has a dependency cycle through "
                f"predecessor {describe(missing)}"
            )
        return tuple(ordered)

    def solve(
        scope: ScopeMemo,
        origin_ns: int,
        completed: dict[int, TimelineEntry],
    ) -> tuple[list[TimelineEntry], list[ScopeTiming]]:
        entries: list[TimelineEntry] = []
        timings: list[ScopeTiming] = []
        available = dict(completed)
        cursors: dict[int, int] = {}
        makespan = 0

        for occurrence in ordered_occurrences(scope):
            ready = origin_ns
            for predecessor, visible in predecessors(occurrence):
                predecessor_entry = available.get(id(visible))
                if predecessor_entry is None or predecessor_entry.expr is not visible:
                    raise AnalysisError(
                        f"{describe(occurrence.expr)}: performance plan invariant: "
                        f"predecessor {describe(predecessor.expr)} materialized in "
                        f"{describe(predecessor.materialized_scope.owner)} has no "
                        f"completed timeline entry visible from "
                        f"{describe(scope.owner)}"
                    )
                ready = max(ready, predecessor_entry.end_ns)

            child_entries: list[TimelineEntry] = []
            child_timings: list[ScopeTiming] = []
            if isinstance(occurrence.expr, Call):
                duration_ns = occurrence.duration_ns * occurrence.replay_count
            else:
                body_scope = structural_memo.scope(occurrence.expr)
                child_entries, child_timings = solve(body_scope, ready, available)
                body_timing = next(
                    timing for timing in child_timings if timing.scope is body_scope
                )
                trips = loop_trip_count(occurrence.expr) * occurrence.replay_count
                duration_ns = body_timing.makespan_ns * trips

            if duration_ns and occurrence.placement:
                held = max(
                    cursors.get(position, 0) for position in occurrence.placement
                )
                if held > ready:
                    ready = held
                    if isinstance(occurrence.expr, GridRegionExpr):
                        body_scope = structural_memo.scope(occurrence.expr)
                        child_entries, child_timings = solve(
                            body_scope, ready, available
                        )

            entries.extend(child_entries)
            timings.extend(child_timings)
            available.update({id(entry.expr): entry for entry in child_entries})
            entry = TimelineEntry(
                expr=occurrence.expr,
                scope=scope,
                duration_ns=duration_ns,
                start_ns=ready,
                end_ns=ready + duration_ns,
            )
            entries.append(entry)
            available[id(entry.expr)] = entry
            if duration_ns:
                for position in occurrence.placement:
                    cursors[position] = entry.end_ns
            makespan = max(makespan, entry.end_ns - origin_ns)

        timings.append(ScopeTiming(scope, makespan))
        return entries, timings

    root_scope = structural_memo.scope(plan.function)
    entries, timings = solve(root_scope, 0, {})
    placement_facts = context.target.get_facts(ParallelCapacityFacts)
    topology = context.module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    waves = -(-topology_extent // placement_facts.parallel_units)
    return _PerformanceTimeline(tuple(entries), tuple(timings), waves)


def attach_performance_records(
    graph: Function,
    timeline: _PerformanceTimeline,
) -> None:
    """Attach public records from flat entries and per-scope makespans."""
    timing_by_scope = {timing.scope: timing for timing in timeline.scopes}
    entry_by_expr = {id(entry.expr): entry for entry in timeline.entries}
    for entry in timeline.entries:
        if not isinstance(entry.expr, Call) or entry.duration_ns == 0:
            continue
        trips = 1
        stride_ns = 0
        if isinstance(entry.scope.owner, GridRegionExpr):
            loop_entry = entry_by_expr[id(entry.scope.owner)]
            body_timing = timing_by_scope[entry.scope]
            if body_timing.makespan_ns:
                trips = loop_entry.duration_ns // body_timing.makespan_ns
            stride_ns = body_timing.makespan_ns if trips > 1 else 0
        attach(
            entry.expr,
            PerformanceMetadata(
                timeline=TimelineMetadata(
                    start_ns=entry.start_ns,
                    end_ns=entry.end_ns,
                    trips=trips,
                    stride_ns=stride_ns,
                )
            ),
        )
    root_timing = next(
        timing for timing in timeline.scopes if timing.scope.owner is graph
    )
    attach(
        graph,
        PerformanceSummaryMetadata(
            timeline=TimelineMetadata(end_ns=root_timing.makespan_ns * timeline.waves),
            waves=timeline.waves,
        ),
    )


def analyze_performance(
    function: Function,
    context: AnalyzeContext,
) -> None:
    """Place every reachable Function's occurrences on a local timeline.

    The buffers this plan keeps live are held to the target's capacities by
    ``memory``, which this family depends on: a time reported here is a time for
    a program whose values have somewhere to sit.
    """
    module, target = context.module, context.target
    placement_facts = target.get_facts(ParallelCapacityFacts)
    topology = module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    for fn in reachable_functions(function):
        memory = get_metadata(fn, MemoryMetadata)
        if memory is None:
            raise AnalysisError(
                f"function {fn.name!r}: performance needs the memory record this "
                "function was never given"
            )
        if memory.allocation is None:
            raise AnalysisError(
                f"function {fn.name!r}: performance reports the time of a program "
                "whose buffers were placed, and this one's machine names no level "
                "to place them against"
            )
        plan = build_performance_plan(fn, context.structural_memo, context)
        timeline = build_timeline(plan, context)
        attach_performance_records(fn, timeline)


__all__ = ["SELECTOR", "analyze_performance"]
