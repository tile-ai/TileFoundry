"""Place modeled work on exact logical participant sets, where it fits.

Each primitive occurrence occupies every logical position named by its result
placement for one CTA-local duration. Dependencies and intersecting participant
sets constrain the intervals, and the buffers those intervals keep live have to
fit the addressable levels at the same time -- a schedule that needs more of a
level than the target has is not a schedule. One model answers both, because
answering them apart lets each one assume the other gave way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, Constant, Expr, Var, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.target import Target
from tilefoundry.target.facts import TARGET_MEMORY_OWNER

from .check import Placement, _call_placements, _result_placement
from .compute_cost import _local_duration_ns
from .errors import AnalysisError
from .facts import (
    ExplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
    ThroughputFacts,
)
from .memory import _base_of, definition_order
from .metadata import (
    BufferAliasMetadata,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    TimelineMetadata,
)
from .walk import (
    attach,
    children,
    describe,
    enclosing_trips,
    loop_scopes,
    loop_trip_count,
    postorder,
    reachable_functions,
)

SELECTOR = "performance"

_ALLOCATED_LEVELS = ("gmem", "smem")
_INT64_LIMIT = 2**62


@dataclass(frozen=True)
class PerformanceOptions:
    """How long the solver may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


@dataclass
class _Occurrence:
    """One primitive occurrence or one structured loop in a local schedule."""

    expr: Call | GridRegionExpr
    placement: Placement
    source_index: int
    duration_ns: int = 0
    trips: int = 1
    predecessors: set[int] = field(default_factory=set)
    body: _Scope | None = None
    start: cp_model.IntVar | None = None
    end: cp_model.IntVar | None = None
    start_ns: int = 0
    end_ns: int = 0


@dataclass
class _Scope:
    """The direct occurrences in one Function or loop body."""

    occurrences: dict[int, _Occurrence]
    external_predecessors: set[int]
    start: cp_model.IntVar
    makespan: cp_model.IntVar
    children: list[_Scope] = field(default_factory=list)
    makespan_ns: int = 0

    @property
    def placement(self) -> Placement:
        return frozenset(
            position
            for occurrence in self.occurrences.values()
            for position in occurrence.placement
        )


@dataclass(frozen=True)
class _Requirement:
    """One buffer the schedule has to keep somewhere, and for how long."""

    base: Expr
    level: ExplicitMemoryLevelFacts
    size_bytes: int
    positions: frozenset[int] | None
    live_start: cp_model.IntVar
    live_end: cp_model.IntVar


@dataclass
class _Problem:
    """Everything one solve decides."""

    model: cp_model.CpModel
    root: _Scope
    requirements: tuple[_Requirement, ...]
    makespan: cp_model.IntVar
    decisions: list[cp_model.IntVar]


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
                f"{describe(expr)}: performance needs the compute-cost record this "
                "call was never given"
            )
        result[id(expr)] = _local_duration_ns(cost, facts, level=level)
    return result


def _horizon(fn: Function, durations: dict[int, int]) -> int:
    """How long the whole function could take if nothing ever overlapped."""
    trips = enclosing_trips(fn.body)
    total = sum(duration * trips.get(key, 1) for key, duration in durations.items())
    if total >= _INT64_LIMIT:
        raise AnalysisError(
            f"function {fn.name!r}: the serial duration bound {total} ns does not fit "
            "the solver's integer range"
        )
    return max(total, 1)


def _producer_ids(expr: Expr, schedulable: set[int]) -> set[int]:
    """Find the nearest schedulable producers beneath a value expression."""
    if id(expr) in schedulable:
        return {id(expr)}
    return {producer for child in children(expr) for producer in _producer_ids(child, schedulable)}


def _overwritten_readers(
    call: Call,
    values: list[Expr],
    users: dict[int, list[Expr]],
    positions: dict[int, int],
    schedulable: set[int],
) -> set[int]:
    """Everything that read the buffer an in-place write is about to overwrite.

    The write reuses those bytes, so it cannot start while anything still needs
    what they held, and def-use ordering does not say this: a reader of the old
    value is not a producer of the new one. Every name for those bytes counts, so
    a view of the buffer is walked too. The write is the group's last authored
    use, which is what made it a write, so every such reader precedes it.
    """
    alias = get_metadata(call, BufferAliasMetadata)
    if alias is None or alias.kind != "update":
        return set()
    base = _base_of(call.args[alias.aliased_operands[0]])
    here = positions[id(call)]
    group = [base, *(expr for expr in values if expr is not call and _base_of(expr) is base)]
    readers: set[int] = set()
    for member in group:
        for reader in _schedulable_users(member, users, schedulable):
            if reader is not call and positions.get(id(reader), -1) < here:
                readers.add(id(reader))
    return readers


def _schedulable_users(
    expr: Expr, users: dict[int, list[Expr]], schedulable: set[int]
) -> list[Expr]:
    """The occurrences that read *expr*, through anything that is not one itself."""
    found: list[Expr] = []
    for user in users.get(id(expr), ()):
        if id(user) in schedulable:
            found.append(user)
        else:
            found.extend(_schedulable_users(user, users, schedulable))
    return found


def _build_scopes(
    model: cp_model.CpModel,
    fn: Function,
    durations: dict[int, int],
    placements: dict[int, Placement],
    horizon: int,
) -> tuple[_Scope, dict[int, _Occurrence], list[cp_model.IntVar]]:
    """Build every scope's interval variables into one model.

    The branching order is innermost scopes first and then authored order: a
    loop's length follows from its body, and schedules of equal length are
    separated by the order the program was written in.
    """
    values = postorder(fn.body)
    source_index = {id(expr): index for index, expr in enumerate(values)}
    parent, scope_of = loop_scopes(fn)
    schedulable = set(scope_of)
    users: dict[int, list[Expr]] = {}
    for expr in values:
        for child in children(expr):
            users.setdefault(id(child), []).append(expr)

    found: dict[int, _Occurrence] = {}
    ordered_starts: list[tuple[int, int, cp_model.IntVar]] = []

    def representative(producer: int, scope: int | None) -> int:
        producer_scope = scope_of[producer]
        if producer_scope == scope:
            return producer
        child = producer_scope
        while child is not None and parent[child] != scope:
            child = parent[child]
        return producer if child is None else child

    def build(scope: int | None, scope_start: cp_model.IntVar, depth: int) -> _Scope:
        direct = [
            expr
            for expr in values
            if isinstance(expr, (Call, GridRegionExpr)) and scope_of[id(expr)] == scope
        ]
        occurrences: dict[int, _Occurrence] = {}
        child_external: dict[int, set[int]] = {}
        bodies: list[_Scope] = []
        for expr in direct:
            index = source_index[id(expr)]
            start = model.NewIntVar(0, horizon, f"s{index}")
            end = model.NewIntVar(0, horizon, f"e{index}")
            ordered_starts.append((depth, index, start))
            model.Add(start >= scope_start)
            if isinstance(expr, Call):
                occurrence = _Occurrence(
                    expr, placements[id(expr)], index, durations[id(expr)], start=start, end=end
                )
                model.Add(end == start + occurrence.duration_ns)
            else:
                body = build(id(expr), start, depth + 1)
                bodies.append(body)
                child_external[id(expr)] = body.external_predecessors
                trips = loop_trip_count(expr)
                occurrence = _Occurrence(
                    expr, body.placement, index, trips=trips, body=body, start=start, end=end
                )
                model.Add(end == start + trips * body.makespan)
            occurrences[id(expr)] = occurrence
            found[id(expr)] = occurrence

        external: set[int] = set()
        for occurrence in occurrences.values():
            expr = occurrence.expr
            operands = expr.args if isinstance(expr, Call) else expr.init_args
            producers = {
                producer for operand in operands for producer in _producer_ids(operand, schedulable)
            }
            if isinstance(expr, GridRegionExpr):
                producers.update(child_external[id(expr)])
            else:
                producers.update(
                    _overwritten_readers(expr, values, users, source_index, schedulable)
                )
            for producer in producers:
                resolved = representative(producer, scope)
                if scope_of[resolved] == scope:
                    occurrence.predecessors.add(resolved)
                else:
                    external.add(resolved)

        _constrain_scope(model, occurrences, scope_start)
        makespan = model.NewIntVar(0, horizon, f"m{scope if scope is not None else 0}")
        if occurrences:
            end_of_scope = model.NewIntVar(0, horizon, f"x{scope if scope is not None else 0}")
            model.AddMaxEquality(
                end_of_scope, [occurrence.end for occurrence in occurrences.values()]
            )
            model.Add(makespan == end_of_scope - scope_start)
        else:
            model.Add(makespan == 0)
        return _Scope(occurrences, external, scope_start, makespan, bodies)

    origin = model.NewIntVar(0, 0, "origin")
    root = build(None, origin, 0)
    decisions = [
        start
        for _depth, _index, start in sorted(
            ordered_starts, key=lambda item: (-item[0], item[1])
        )
    ]
    return root, found, decisions


def _constrain_scope(
    model: cp_model.CpModel,
    occurrences: dict[int, _Occurrence],
    scope_start: cp_model.IntVar,
) -> None:
    """Order one scope's occurrences and keep its participants to one at a time."""
    intervals_by_position: dict[int, list[cp_model.IntervalVar]] = {}
    for occurrence in occurrences.values():
        for predecessor in occurrence.predecessors:
            model.Add(occurrence.start >= occurrences[predecessor].end)
        if isinstance(occurrence.expr, Call) and occurrence.duration_ns == 0:
            if occurrence.predecessors:
                model.AddMaxEquality(
                    occurrence.start,
                    [occurrences[predecessor].end for predecessor in occurrence.predecessors],
                )
            else:
                model.Add(occurrence.start == scope_start)
            continue
        span = (
            occurrence.duration_ns
            if isinstance(occurrence.expr, Call)
            else occurrence.trips * occurrence.body.makespan
        )
        interval = model.NewIntervalVar(
            occurrence.start, span, occurrence.end, f"i{occurrence.source_index}"
        )
        for position in occurrence.placement:
            intervals_by_position.setdefault(position, []).append(interval)
    for intervals in intervals_by_position.values():
        model.AddNoOverlap(intervals)


def _live_bounds(
    lifetime_start: Expr,
    lifetime_end: Expr,
    occurrences: dict[int, _Occurrence],
    scope_of: dict[int, int | None],
    parent: dict[int, int | None],
    root: _Scope,
) -> tuple[cp_model.IntVar, cp_model.IntVar]:
    """When a buffer is first needed and last read, as this model's variables.

    Values a loop produces and consumes within one trip are bound to that trip's
    own variables, so later trips reuse whatever address the first one got. A
    value that crosses the loop boundary is bound to the loop as a whole, which
    is how long it really has to stay somewhere.
    """
    start_scope = _enclosing(lifetime_start, scope_of)
    end_scope = _enclosing(lifetime_end, scope_of)
    common = _common_scope(start_scope, end_scope, parent)
    first = _lifted(lifetime_start, common, scope_of, occurrences)
    last = _lifted(lifetime_end, common, scope_of, occurrences)
    return (
        root.start if first is None else first.start,
        root.makespan if last is None else last.end,
    )


def _enclosing(expr: Expr, scope_of: dict[int, int | None]) -> int | None:
    return scope_of.get(id(expr))


def _common_scope(
    left: int | None, right: int | None, parent: dict[int, int | None]
) -> int | None:
    """The innermost loop that contains both, or the Function body."""
    chain: list[int | None] = []
    walk = left
    while True:
        chain.append(walk)
        if walk is None:
            break
        walk = parent[walk]
    walk = right
    while walk is not None and walk not in chain:
        walk = parent[walk]
    return walk


def _lifted(
    expr: Expr,
    scope: int | None,
    scope_of: dict[int, int | None],
    occurrences: dict[int, _Occurrence],
) -> _Occurrence | None:
    """The occurrence in *scope* that stands for *expr*'s time.

    A value inside a loop is stood for by that loop once the question is asked
    from outside it, which is how a buffer that crosses the boundary comes to be
    live for the whole loop rather than for one trip.
    """
    key = id(expr)
    while key in scope_of and scope_of[key] != scope:
        owner = scope_of[key]
        if owner is None:
            return None
        key = owner
    return occurrences.get(key)


def _requirements(
    fn: Function,
    record: MemoryMetadata,
    hierarchy: MemoryHierarchyFacts,
    placements: dict[int, Placement],
    occurrences: dict[int, _Occurrence],
    root: _Scope,
    selected: str,
) -> tuple[_Requirement, ...]:
    """Every gmem/smem buffer this schedule has to hold, and when."""
    order = definition_order(fn)
    parent, scope_of = loop_scopes(fn)
    found: list[_Requirement] = []
    for item in record.lifetimes:
        level = hierarchy.explicit(item.level)
        if level is None or level.name not in _ALLOCATED_LEVELS:
            continue
        if not 0 <= item.defined_at < len(order):
            raise AnalysisError(
                f"function {fn.name!r}: lifetime {item.binding!r} names definition "
                f"{item.defined_at}, which is outside this function's value order"
            )
        base = order[item.defined_at]
        last = order[min(item.last_used_at, len(order) - 1)]
        live_start, live_end = _live_bounds(base, last, occurrences, scope_of, parent, root)
        if item.persistent:
            live_start, live_end = root.start, root.makespan
        found.append(
            _Requirement(
                base=base,
                level=level,
                size_bytes=item.bytes,
                positions=_scope_positions(fn, base, level, placements, selected),
                live_start=live_start,
                live_end=live_end,
            )
        )
    return tuple(found)


def _scope_positions(
    fn: Function,
    base: Expr,
    level: ExplicitMemoryLevelFacts,
    placements: dict[int, Placement],
    selected: str,
) -> frozenset[int] | None:
    """Which capacity domains hold an instance of this buffer.

    ``None`` is the whole target: one allocation everybody shares. Otherwise the
    level is owned per unit of the level being analysed, and the value sits in
    exactly the positions its placement names.
    """
    if level.owner == TARGET_MEMORY_OWNER:
        return None
    if level.owner != selected or level.scope != level.owner:
        raise AnalysisError(
            f"function {fn.name!r}: {level.name!r} is owned per {level.owner!r} and "
            f"measured per {level.scope!r}, which this analysis cannot project onto "
            f"the {selected!r} level it was asked about"
        )
    placement = placements.get(id(base))
    if placement is None:
        raise AnalysisError(
            f"function {fn.name!r}: a {level.name!r} buffer has no {selected!r} "
            "position to be measured against"
        )
    return placement


def _add_capacity(
    model: cp_model.CpModel, requirements: tuple[_Requirement, ...], horizon: int
) -> None:
    """Hold every capacity domain to one address range per live buffer.

    Domains that hold the same buffers are the same constraint written twice, so
    one of them stands for the rest. Domains that hold different buffers, or the
    same buffers at different sizes, each get their own.

    No offset needs to look past the buffers stacked end to end: a level with
    room for all of them at once has room for any one of them below that mark.
    """
    by_domain: dict[tuple[str, object], list[_Requirement]] = {}
    for requirement in requirements:
        positions = (
            (None,) if requirement.positions is None else sorted(requirement.positions)
        )
        for position in positions:
            by_domain.setdefault((requirement.level.name, position), []).append(requirement)

    seen: set[tuple] = set()
    for (name, _position), domain in by_domain.items():
        signature = (name, tuple(sorted(id(item.base) for item in domain)))
        if signature in seen:
            continue
        seen.add(signature)
        capacity = domain[0].level.capacity_bytes
        if capacity is None or capacity <= 0:
            raise AnalysisError(
                f"the target states no usable capacity for {name!r}, so a program "
                "that places values there cannot be shown to fit"
            )
        reach = min(capacity, sum(item.size_bytes for item in domain))
        addresses: list[cp_model.IntervalVar] = []
        lives: list[cp_model.IntervalVar] = []
        for index, requirement in enumerate(domain):
            if requirement.size_bytes > capacity:
                raise AnalysisError(
                    f"a {name!r} buffer needs {requirement.size_bytes} B, more than "
                    f"the {capacity} B the target states for that level"
                )
            offset = model.NewIntVar(0, reach - requirement.size_bytes, f"o{name}{index}")
            addresses.append(
                model.NewIntervalVar(
                    offset,
                    requirement.size_bytes,
                    offset + requirement.size_bytes,
                    f"a{name}{index}",
                )
            )
            span = model.NewIntVar(0, horizon, f"d{name}{index}")
            model.Add(span == requirement.live_end - requirement.live_start)
            lives.append(
                model.NewIntervalVar(
                    requirement.live_start, span, requirement.live_end, f"l{name}{index}"
                )
            )
        model.AddNoOverlap2D(addresses, lives)


def _solve(problem: _Problem, options: PerformanceOptions) -> str:
    """Minimize the makespan, and say whether the answer was proved."""
    problem.model.Minimize(problem.makespan)
    if problem.decisions:
        problem.model.AddDecisionStrategy(
            problem.decisions, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
        )
    solver = cp_model.CpSolver()
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.num_search_workers = options.workers
    solver.parameters.random_seed = options.random_seed
    solver.parameters.max_time_in_seconds = options.timeout_seconds
    status = solver.Solve(problem.model)
    if status == cp_model.INFEASIBLE:
        raise AnalysisError(
            "no timeline places this program's buffers within the target's "
            "addressable capacities"
        )
    if status == cp_model.MODEL_INVALID:
        raise AnalysisError("the performance model is not a valid solver problem")
    if status == cp_model.UNKNOWN:
        raise AnalysisError(
            "the performance solver found no timeline within its time limit"
        )
    _read_back(solver, problem.root)
    return "optimal" if status == cp_model.OPTIMAL else "feasible"


def _read_back(solver: cp_model.CpSolver, scope: _Scope) -> None:
    """Copy the solved times onto the occurrences they belong to."""
    scope.makespan_ns = solver.Value(scope.makespan)
    for occurrence in scope.occurrences.values():
        occurrence.start_ns = solver.Value(occurrence.start)
        occurrence.end_ns = solver.Value(occurrence.end)
    for child in scope.children:
        _read_back(solver, child)


def _records(
    scope: _Scope,
    *,
    trips: int = 1,
    stride_ns: int = 0,
) -> dict[int, PerformanceMetadata]:
    """Materialize absolute first-trip intervals from a solved scope tree."""
    result: dict[int, PerformanceMetadata] = {}
    for key, occurrence in scope.occurrences.items():
        if isinstance(occurrence.expr, Call):
            if occurrence.duration_ns == 0:
                continue
            result[key] = PerformanceMetadata(
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
    """Place every reachable Function's occurrences on a local timeline."""
    placement_facts = target.get_facts(ParallelCapacityFacts)
    throughput = target.get_facts(ThroughputFacts)
    hierarchy = target.get_facts(MemoryHierarchyFacts)
    settings = options if isinstance(options, PerformanceOptions) else PerformanceOptions()
    topology = module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    waves = -(-topology_extent // placement_facts.parallel_units)
    for fn in reachable_functions(function):
        placements = _call_placements(module, fn, placement_facts.topology, throughput)
        durations = _durations(fn, throughput, placement_facts.topology)
        memory = get_metadata(fn, MemoryMetadata)
        if memory is None:
            raise AnalysisError(
                f"function {fn.name!r}: performance needs the memory record this "
                "function was never given"
            )
        model = cp_model.CpModel()
        horizon = _horizon(fn, durations)
        root, occurrences, decisions = _build_scopes(
            model, fn, durations, placements, horizon
        )
        requirements = _requirements(
            fn,
            memory,
            hierarchy,
            _buffer_placements(fn, placements, placement_facts.topology, module),
            occurrences,
            root,
            placement_facts.topology,
        )
        _add_capacity(model, requirements, horizon)
        status = _solve(
            _Problem(model, root, requirements, root.makespan, decisions), settings
        )
        records = _records(root)
        for expr in postorder(fn.body):
            record = records.get(id(expr))
            if record is not None:
                attach(expr, record)
        attach(
            fn,
            PerformanceSummaryMetadata(
                timeline=TimelineMetadata(end_ns=root.makespan_ns * waves),
                waves=waves,
                solver_status=status,
            ),
        )


def _buffer_placements(
    fn: Function,
    placements: dict[int, Placement],
    selected: str,
    module: Module,
) -> dict[int, Placement]:
    """Where every value that can own a buffer lives, parameters included."""
    resolved = dict(placements)
    topology = module.resolve_topology(selected)
    for expr in (*fn.params, *postorder(fn.body)):
        if id(expr) in resolved or not isinstance(expr, (Var, Constant)):
            continue
        try:
            resolved[id(expr)] = _result_placement(expr.type, topology)
        except AnalysisError:
            continue
    return resolved


__all__ = ["PerformanceOptions", "SELECTOR", "analyze_performance"]
