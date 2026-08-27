"""Place modeled work on exact logical participant sets, in the authored order.

Each occurrence holds every position the Mesh it was authored inside names, for
one CTA-local duration. A position runs one occurrence at a time and an
occurrence waits for what it reads, so the program's own placement says what
overlaps. Nothing is searched. Where the buffers sit belongs to ``memory``.

A ``_Chain`` names one scope by the loops its occurrences belong to, outermost
first; an occurrence's ``lifted`` count is how many runs of one value it stands
for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.target import Target

from .check import Placement, _call_placements
from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import (
    ParallelCapacityFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    TimelineMetadata,
    TrafficMetadata,
)
from .walk import (
    attach,
    children,
    collect_exprs,
    describe,
    loop_repeated_values,
    loop_trip_count,
    reachable_functions,
)

SELECTOR = "performance"

_Chain = tuple[int, ...]


@dataclass
class _Occurrence:
    """One primitive occurrence or one structured loop in a local schedule."""

    expr: Call | GridRegionExpr
    placement: Placement
    source_index: int
    duration_ns: int = 0
    trips: int = 1
    lifted: int = 1
    predecessors: set[int] = field(default_factory=set)
    body: _Scope | None = None
    start_ns: int = 0
    end_ns: int = 0


@dataclass
class _Scope:
    """The direct occurrences in one Function or loop body."""

    occurrences: dict[int, _Occurrence]
    children: list[_Scope] = field(default_factory=list)
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
    services: PerformanceServiceFacts,
    level: str,
) -> dict[int, int]:
    """Return one CTA-local duration for every primitive occurrence."""
    result: dict[int, int] = {}
    for expr in collect_exprs(fn.body):
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


def _producer_ids(expr: Expr, schedulable: set[int]) -> set[int]:
    """Find the nearest schedulable producers beneath a value expression."""
    if id(expr) in schedulable:
        return {id(expr)}
    return {producer for child in children(expr) for producer in _producer_ids(child, schedulable)}








def _variance_chains(fn: Function) -> tuple[dict[int, _Chain], dict[int, GridRegionExpr]]:
    """Which loops each schedulable value belongs to, from the outside in.

    A loop holds a value only when the value reads that loop's induction
    variable or one of its carried arguments, which is the rule the cost
    families count repetition by. Every loop that repeats a value also contains
    it lexically, so the loops one value belongs to nest in one another and read
    as a chain. A loop that repeats a value's neighbours but not the value is
    simply not on that chain.
    """
    values = collect_exprs(fn.body)
    order = {id(expr): index for index, expr in enumerate(values)}
    loops = {id(expr): expr for expr in values if isinstance(expr, GridRegionExpr)}
    repeated = {key: loop_repeated_values(loop) for key, loop in loops.items()}
    chains: dict[int, _Chain] = {}
    for expr in values:
        if not isinstance(expr, (Call, GridRegionExpr)):
            continue
        owners = [key for key, marked in repeated.items() if id(expr) in marked]
        chains[id(expr)] = tuple(sorted(owners, key=lambda key: -order[key]))
    return chains, loops


def _placement_plan(
    fn: Function,
) -> tuple[dict[int, _Chain], dict[int, int], dict[_Chain, GridRegionExpr]]:
    """Where each occurrence sits, and how many runs of it that one stands for.

    One node per authored loop, so work that ran together is never split into two
    nodes that then wait for each other. A value the loop repeats without
    changing lifts out to the deepest node it does belong to, carrying the runs
    it still owes below as a count, and so stands for exactly the repetition the
    cost families charged.
    """
    values = collect_exprs(fn.body)
    order = {id(expr): index for index, expr in enumerate(values)}
    chains, loops = _variance_chains(fn)
    nodes: dict[_Chain, GridRegionExpr | None] = {(): None}

    def host_of(chain: _Chain) -> _Chain:
        found: _Chain = ()
        for size in range(1, len(chain) + 1):
            if chain[:size] in nodes:
                found = chain[:size]
        return found

    hosts: dict[int, _Chain] = {}
    replays: dict[int, int] = {}

    def settle(key: int) -> _Chain:
        host = host_of(chains[key])
        hosts[key] = host
        count = 1
        for owner in chains[key][len(host) :]:
            count *= loop_trip_count(loops[owner])
        replays[key] = count
        return host

    for loop in sorted(loops.values(), key=lambda expr: -order[id(expr)]):
        nodes[(*settle(id(loop)), id(loop))] = loop
    for expr in values:
        if isinstance(expr, Call):
            settle(id(expr))
    bodies = {chain: loop for chain, loop in nodes.items() if loop is not None}
    return hosts, replays, bodies


def _schedule(
    fn: Function,
    durations: dict[int, int],
    placements: dict[int, Placement],
) -> _Scope:
    """Lay every occurrence on one local timeline, in the order it was written.

    Each participant runs one thing at a time, so an occurrence waits for the
    last occurrence naming any position it names, and for every value it reads.
    Nothing is reordered: the program says what is independent through where it
    places the work, and this reports how long that program takes rather than
    how long a better ordering would.

    A loop body is laid out against its own origin and then moved to where the
    loop starts, which makes a body interval the first of its ``trips``.
    """
    values = collect_exprs(fn.body)
    source_index = {id(expr): index for index, expr in enumerate(values)}
    hosts, replays, _bodies = _placement_plan(fn)
    schedulable = set(hosts)
    users: dict[int, list[Expr]] = {}
    for expr in values:
        for child in children(expr):
            users.setdefault(id(child), []).append(expr)

    direct: dict[_Chain, list[Call | GridRegionExpr]] = {}
    for expr in values:
        if id(expr) in hosts:
            direct.setdefault(hosts[id(expr)], []).append(expr)

    def representative(producer: int, scope: _Chain) -> int | None:
        host = hosts[producer]
        if host == scope:
            return producer
        if len(host) > len(scope) and host[: len(scope)] == scope:
            return host[len(scope)]
        return None

    def held_under(scope: _Chain) -> Placement:
        """Which participants run anything in one scope, before it is laid out.

        A loop occupies whatever its body occupies, and a body is laid out from
        where the loop starts -- so the two cannot both wait for each other. The
        occupation is a fact about the program's placement rather than about its
        timing, and is read straight off the occurrences beneath.
        """
        found: set[int] = set()
        for expr in direct.get(scope, ()):
            if isinstance(expr, Call):
                found |= placements[id(expr)]
            else:
                found |= held_under((*scope, id(expr)))
        return frozenset(found)

    def build(scope: _Chain, origin_ns: int) -> _Scope:
        occurrences: dict[int, _Occurrence] = {}
        children_scopes: list[_Scope] = []
        cursors: dict[int, int] = {}
        makespan = 0
        for expr in sorted(direct.get(scope, ()), key=lambda item: source_index[id(item)]):
            index = source_index[id(expr)]
            runs = replays[id(expr)]
            here = (*scope, id(expr))
            occurrence = _Occurrence(
                expr,
                placements[id(expr)] if isinstance(expr, Call) else held_under(here),
                index,
                durations[id(expr)] * runs if isinstance(expr, Call) else 0,
                trips=1 if isinstance(expr, Call) else loop_trip_count(expr) * runs,
                lifted=runs,
            )
            occurrences[id(expr)] = occurrence

            operands = expr.args if isinstance(expr, Call) else expr.init_args
            producers = {
                producer for operand in operands for producer in _producer_ids(operand, schedulable)
            }
            for producer in producers:
                resolved = representative(producer, scope)
                if resolved is not None and resolved != id(expr):
                    occurrence.predecessors.add(resolved)

            ready = max(
                (occurrences[key].end_ns for key in occurrence.predecessors),
                default=origin_ns,
            )
            if not isinstance(expr, Call):
                occurrence.body = build(here, ready)
                occurrence.duration_ns = (
                    occurrence.body.makespan_ns * occurrence.trips
                )
            if occurrence.duration_ns:
                held = max(cursors.get(position, 0) for position in occurrence.placement)
                if held > ready:
                    ready = held
                    if occurrence.body is not None:
                        occurrence.body = build(here, ready)
            if occurrence.body is not None:
                children_scopes.append(occurrence.body)
            occurrence.start_ns = ready
            occurrence.end_ns = ready + occurrence.duration_ns
            if occurrence.duration_ns:
                for position in occurrence.placement:
                    cursors[position] = occurrence.end_ns
            makespan = max(makespan, occurrence.end_ns - origin_ns)

        return _Scope(occurrences, children_scopes, makespan)

    return build((), 0)


def _records(
    scope: _Scope,
    *,
    trips: int = 1,
    stride_ns: int = 0,
) -> dict[int, PerformanceMetadata]:
    """Materialize absolute first-trip intervals from a solved scope tree."""
    result: dict[int, PerformanceMetadata] = {}
    for occurrence in scope.occurrences.values():
        if isinstance(occurrence.expr, Call):
            if occurrence.duration_ns == 0:
                continue
            result[id(occurrence.expr)] = PerformanceMetadata(
                timeline=TimelineMetadata(
                    start_ns=occurrence.start_ns,
                    end_ns=occurrence.end_ns,
                    trips=trips,
                    stride_ns=stride_ns if trips > 1 else 0,
                )
            )
            continue
        if occurrence.body is not None:
            result.update(
                _records(
                    occurrence.body,
                    trips=occurrence.trips,
                    stride_ns=occurrence.body.makespan_ns,
                )
            )
    return result


def analyze_performance(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Place every reachable Function's occurrences on a local timeline.

    The buffers this plan keeps live are held to the target's capacities by
    ``memory``, which this family depends on: a time reported here is a time for
    a program whose values have somewhere to sit.
    """
    placement_facts = target.get_facts(ParallelCapacityFacts)
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)
    topology = module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    waves = -(-topology_extent // placement_facts.parallel_units)
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
        placements = _call_placements(module, fn, placement_facts.topology)
        durations = _durations(fn, throughput, services, placement_facts.topology)
        root = _schedule(fn, durations, placements)
        records = _records(root)
        for expr in collect_exprs(fn.body):
            record = records.get(id(expr))
            if record is not None:
                attach(expr, record)
        attach(
            fn,
            PerformanceSummaryMetadata(
                timeline=TimelineMetadata(end_ns=root.makespan_ns * waves),
                waves=waves,
            ),
        )


__all__ = ["SELECTOR", "analyze_performance"]
