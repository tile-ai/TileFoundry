"""Measure function residency against the target memory hierarchy, and place it.

Lifetime order comes from authored IR and residency is projected to each
explicit level's owner. A single value exceeding a level is an error; peak and
cache overflow are advisories.

``_ALLOCATED_LEVELS`` are then addressed: buffers live at once get byte ranges of
their own within the stated capacity, and authored order already fixes the
lifetimes. A program whose buffers cannot be placed is refused here, so a
recorded allocation is always one that holds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import isl
from ortools.sat.python import cp_model

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Tuple,
    Var,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import Type, local_type_of, tensor_bytes
from tilefoundry.ir.types.shard import Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import Target, UnsupportedCapabilityError
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
)
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope

from .buffer_plan import build_buffer_plan
from .check import Placement, _call_placements, _result_placement
from .errors import AnalysisError
from .facts import (
    TARGET_MEMORY_OWNER,
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    ParallelCapacityFacts,
)
from .footprint import loop_footprints
from .metadata import (
    AllocationMetadata,
    BufferAliasMetadata,
    BufferAllocationMetadata,
    BufferFootprint,
    BufferRef,
    ComputeCostMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    MemoryMetadata,
    TrafficBytes,
    ValueLifetime,
)
from .traffic import TrafficMetadata, lower_traffic
from .walk import (
    attach,
    bytes_by_storage,
    children,
    enclosing_trips,
    postorder,
    reachable_functions,
    tensor_types,
)

SELECTOR = "memory"
_UMAT_LEVEL = str(StorageKind.RMEM)

_ALLOCATED_LEVELS = ("gmem", "smem")


@dataclass(frozen=True)
class _Residency:
    """One value's claim on one level, over one span of the definition order."""

    binding: str
    level: str
    bytes: int
    defined_at: int
    last_used_at: int
    persistent: bool


@dataclass(frozen=True)
class _CachePressure:
    """One authored loop's device-wide access against one implicit cache."""

    cache_level: str
    backing_level: str
    device_bytes: int
    capacity_bytes: int | None
    status: str


def _is_view(expr: Expr) -> bool:
    """Whether *expr* lives in a buffer another value already owns.

    The conclusion is compute-cost's, from a proof against the operation's own
    addresses. An operation that could have forwarded but did not prove it is
    recorded as producing, and allocates here like anything else.
    """
    alias = get_metadata(expr, BufferAliasMetadata)
    return alias is not None and alias.kind in ("forward", "update")


def _carried_origins(fn: Function) -> dict[int, Expr]:
    """Where a loop-shaped value's bytes come from.

    A loop that updates a buffer names the value it is updating once per
    iteration, and those names are the loop's own. No iteration allocates, so
    the buffer a carried value stands for is the one the loop was entered with.

    A loop's own result is not here: it is what its last iteration yielded, one
    field per carried value, which is a structure and not a rename.
    """
    found: dict[int, Expr] = {}
    if fn.body is None:
        return found
    for expr in postorder(fn.body):
        if not isinstance(expr, GridRegionExpr):
            continue
        for carried, init in zip(expr.carried_args, expr.init_args, strict=False):
            found[id(carried)] = init
    return found


def _base_of(
    expr: Expr,
    seen: frozenset[int] = frozenset(),
    carried: dict[int, Expr] | None = None,
) -> Expr:
    """Follow the proven operand edges to the value that owns the bytes."""
    if id(expr) in seen:
        raise AnalysisError(f"{binding_name(expr) or 'a value'} aliases itself")
    if carried is not None and id(expr) in carried:
        return _base_of(carried[id(expr)], seen | {id(expr)}, carried)
    alias = get_metadata(expr, BufferAliasMetadata)
    if alias is None or not alias.aliased_operands:
        return expr
    return _base_of(expr.args[alias.aliased_operands[0]], seen | {id(expr)}, carried)


def _label(expr: Expr, position: int) -> str:
    """How a value is named in a lifetime record.

    A parameter carries its own name; a body value carries the authored binding.
    A value with neither is labelled by its place in the order rather than by its
    source span, because the record travels and a path from the authoring machine
    means nothing elsewhere.
    """
    if isinstance(expr, Var):
        return expr.name
    return binding_name(expr) or f"<value {position}>"


def _unique_labels(order: list[Expr]) -> dict[int, str]:
    """Assign one unambiguous label per value in definition order.

    The parser stamps one authored binding on nested right-hand-side values, so
    repeated names receive the printer's numeric suffix while the first remains
    bare. Assign across the complete order so emitted memory levels cannot
    renumber values. This is temporary until scoped value identity replaces the
    parser's shared binding labels.
    """
    taken: set[str] = set()
    labels: dict[int, str] = {}
    for position, expr in enumerate(order):
        base = _label(expr, position)
        name, suffix = base, 2
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        taken.add(name)
        labels[id(expr)] = name
    return labels


def definition_order(fn: Function) -> list[Expr]:
    """Every value of *fn* in the one order its lifetimes are indexed by.

    Parameters come first: the function did not produce them, so they are
    already resident when it is entered.
    """
    return [
        *fn.params,
        *(expr for expr in postorder(fn.body) if isinstance(expr, (Call, Constant))),
    ]


def _residencies(
    fn: Function,
    *,
    facts: MemoryHierarchyFacts,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> tuple[tuple[_Residency, ...], int]:
    """Every allocation resident in *fn*, with the length of its value order.

    Parameters are part of the order and start at its beginning: the function
    did not produce them, so they are already resident when it is entered. A
    function cannot reclaim storage its caller owns, so every parameter stays
    resident past its last reader for the whole function.
    """
    order = definition_order(fn)
    position = {id(expr): index for index, expr in enumerate(order)}
    last_use = dict(position)
    for consumer in order:
        for child in children(consumer):
            if id(child) in last_use:
                last_use[id(child)] = max(
                    last_use[id(child)], position[id(consumer)]
                )
    if fn.body is not None and id(fn.body) in last_use:
        last_use[id(fn.body)] = len(order) - 1
    for expr in reversed(order):
        if not _is_view(expr):
            continue
        base = _base_of(expr)
        if id(base) in last_use:
            last_use[id(base)] = max(last_use[id(base)], last_use[id(expr)])

    labels = _unique_labels(order)

    result: list[_Residency] = []
    for expr in order:
        if _is_view(expr):
            continue
        persistent = isinstance(expr, Var)
        for storage in bytes_by_storage(expr.type):
            declared = facts.explicit(storage)
            type_ = (
                expr.type
                if declared is None
                else _type_at_owner(
                    expr.type,
                    owner=declared.owner,
                    topology_levels=topology_levels,
                    topologies=topologies,
                )
            )
            amount = bytes_by_storage(type_)[storage]
            result.append(
                _Residency(
                    binding=labels[id(expr)],
                    level=storage,
                    bytes=amount,
                    defined_at=position[id(expr)],
                    last_used_at=(
                        len(order) - 1 if persistent else last_use[id(expr)]
                    ),
                    persistent=persistent,
                )
            )
    return tuple(result), len(order)


def _type_at_owner(
    type_: Type,
    *,
    owner: str,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> Type:
    """Project *type_* through every declared split no finer than *owner*."""
    if owner == TARGET_MEMORY_OWNER:
        return type_
    try:
        owner_index = topology_levels.index(owner)
    except ValueError:
        available = ", ".join((TARGET_MEMORY_OWNER, *topology_levels))
        raise ValueError(
            f"memory owner {owner!r} is not declared by the target; "
            f"available owners are {available}"
        ) from None
    projection_levels = tuple(
        topology.name
        for topology in topologies
        if topology_levels.index(topology.name) <= owner_index
    )
    if not projection_levels:
        return type_
    return local_type_of(type_, level=projection_levels[-1], topologies=topologies)


def _peaks(
    residencies: tuple[_Residency, ...], length: int
) -> dict[str, int]:
    """The largest simultaneous claim on each level over the whole order."""
    peaks: dict[str, int] = {}
    for index in range(max(length, 1)):
        live: dict[str, int] = {}
        for item in residencies:
            if item.defined_at <= index <= item.last_used_at:
                live[item.level] = live.get(item.level, 0) + item.bytes
        for level, amount in live.items():
            peaks[level] = max(peaks.get(level, 0), amount)
    return peaks


def _function_traffic(fn: Function) -> tuple[tuple[str, TrafficBytes], ...]:
    """Multiplicity-aware traffic from the compute-cost root record."""
    record = get_metadata(fn, ComputeCostMetadata)
    if record is None:
        raise AnalysisError(
            f"function {fn.name!r}: memory needs the compute-cost root record "
            "this function was never given"
        )
    return record.traffic


def _explicit_footprint(
    fn: Function,
    residencies: tuple[_Residency, ...],
    peaks: dict[str, int],
    persistent: dict[str, int],
    facts: MemoryHierarchyFacts,
) -> tuple[LevelFootprint, ...]:
    """One footprint row per level the function places values in.

    A level the target does not declare is still reported, with no capacity: the
    program used it, and dropping the row would hide that.
    """
    rows: list[LevelFootprint] = []
    for level in sorted(peaks):
        declared = facts.explicit(level)
        row = LevelFootprint(
            level=level,
            peak_bytes=peaks[level],
            persistent_bytes=persistent.get(level, 0),
            capacity_bytes=None if declared is None else declared.capacity_bytes,
        )
        oversized = next(
            (
                item
                for item in residencies
                if item.level == level
                and row.capacity_bytes is not None
                and item.bytes > row.capacity_bytes
            ),
            None,
        )
        if oversized is not None:
            raise AnalysisError(
                f"function {fn.name!r}: value {oversized.binding!r} needs "
                f"{oversized.bytes} B in {level}, which exceeds the "
                f"{row.capacity_bytes} B the target states for that level"
            )
        rows.append(row)
    return tuple(rows)


def _explicit_peak_advisories(
    facts: MemoryHierarchyFacts, peaks: dict[str, int]
) -> tuple[str, ...]:
    """Explicit-level peaks that exceed capacity under this walk's order."""
    notes: list[str] = []
    for level in sorted(peaks):
        declared = facts.explicit(level)
        if (
            declared is not None
            and declared.capacity_bytes is not None
            and peaks[level] > declared.capacity_bytes
        ):
            notes.append(
                f"{level} peak is {peaks[level]} B under this walk's value "
                f"order, exceeding its {declared.capacity_bytes} B capacity; "
                "the peak is order-dependent and is not a bound over schedules"
            )
    return tuple(notes)


def _implicit_capacity(
    level: str, facts: MemoryHierarchyFacts, peaks: dict[str, int]
) -> int | None:
    """How much of an implicit level is left for the traffic it fronts.

    A level that divides a physical block with an addressable one only gets what
    that addressable level did not take, so its usable capacity depends on the
    program rather than being a constant of the machine.
    """
    declared = facts.implicit(level)
    capacity = None if declared is None else declared.capacity_bytes
    for peer, shared_bytes in facts.capacity_sharers(level):
        if shared_bytes is None:
            continue
        remaining = shared_bytes - peaks.get(peer, 0)
        capacity = remaining if capacity is None else min(capacity, remaining)
    return capacity


def cache_pressure(
    record: LoopFootprintMetadata,
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
) -> tuple[_CachePressure, ...]:
    """Compare one loop's device-wide explicit accesses with their caches.

    The comparison is made only when the cache and the level it backs are stated
    per the same topology scope. A per-SM capacity set against a whole-device
    footprint would exceed it for almost any program, which reads as a finding
    while saying nothing: the per-SM share of that footprint is not known here.
    """
    rows: list[_CachePressure] = []
    for level in facts.implicit_levels:
        backing_name = facts.backing_level(level.name)
        backing = facts.explicit(backing_name)
        if backing is None or backing.scope != level.scope:
            continue
        accesses = tuple(
            item for item in record.footprints if item.level == backing_name
        )
        if not accesses or any(item.device_bytes < item.bytes for item in accesses):
            continue
        working_set = sum(item.device_bytes for item in accesses)
        capacity = _implicit_capacity(level.name, facts, peaks)
        if capacity is None:
            status = "unknown"
        elif working_set > capacity:
            status = "exceeds"
        elif record.known:
            status = "fits"
        else:
            status = "lower-bound"
        rows.append(
            _CachePressure(
                cache_level=level.name,
                backing_level=backing_name,
                device_bytes=working_set,
                capacity_bytes=capacity,
                status=status,
            )
        )
    return tuple(rows)


def _residency_advisory(
    loop: GridRegionExpr,
    record: LoopFootprintMetadata,
    pressure: _CachePressure,
    facts: MemoryHierarchyFacts,
) -> str | None:
    """Report one loop whose access footprint exceeds a same-scope cache."""
    if pressure.status != "exceeds" or pressure.capacity_bytes is None:
        return None
    level = facts.implicit(pressure.cache_level)
    if level is None:
        return None
    amount = (
        str(pressure.device_bytes)
        if record.known
        else f"at least {pressure.device_bytes}"
    )
    return (
        f"{pressure.cache_level} holds {pressure.capacity_bytes} B per {level.scope}, "
        f"so the {pressure.backing_level} access footprint of {amount} B in loop "
        f"{loop.induction_var.name!r} will not stay resident"
    )


def _division_advisory(
    level: ImplicitMemoryLevelFacts,
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
) -> str | None:
    """How much of a shared block this program leaves *level*.

    This says nothing about a working set, so it needs no matching scope: it is a
    statement about the block itself, and it is worth making only once some peer
    has actually claimed part of it.
    """
    for peer, shared_bytes in facts.capacity_sharers(level.name):
        claimed = peaks.get(peer, 0)
        if shared_bytes is None or not claimed:
            continue
        remaining = shared_bytes - claimed
        return (
            f"{peer} claims {claimed} B of the {shared_bytes} B block it divides "
            f"with {level.name}, leaving {level.name} {remaining} B"
        )
    return None


def _advisories(
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
    loops: tuple[tuple[GridRegionExpr, LoopFootprintMetadata], ...],
) -> tuple[str, ...]:
    """Order and cache findings, none of which invalidate the program."""
    notes = list(_explicit_peak_advisories(facts, peaks))
    for level in facts.implicit_levels:
        note = _division_advisory(level, facts, peaks)
        if note is not None:
            notes.append(note)
    for loop, record in loops:
        notes.extend(
            note
            for pressure in cache_pressure(record, facts, peaks)
            if (note := _residency_advisory(loop, record, pressure, facts)) is not None
        )
    return tuple(notes)


@dataclass(frozen=True)
class MemoryOptions:
    """How long the placement may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


class _NotProjectable(Exception):
    """No level of this program can be projected onto one that holds addresses."""


@dataclass(frozen=True)
class _Rectangle:
    """One buffer's byte size and the span of the program it is live across."""

    binding: str
    size_bytes: int
    first: int
    last: int


def _domains(
    fn: Function,
    record: MemoryMetadata,
    facts: MemoryHierarchyFacts,
    placements: dict[int, Placement],
    order: list[Expr],
    selected: str,
) -> dict[tuple[str, object], list[_Rectangle]]:
    """Every capacity domain that has to hold something, and what it holds.

    A level the target owns is one allocation the whole program shares. A level
    owned per unit of the level being analysed has one allocation per unit, and
    a buffer belongs to each unit its placement names.
    """
    found: dict[tuple[str, object], list[_Rectangle]] = {}
    for item in record.lifetimes:
        level = facts.explicit(item.level)
        if level is None or level.name not in _ALLOCATED_LEVELS:
            continue
        if not 0 <= item.defined_at < len(order):
            raise AnalysisError(
                f"function {fn.name!r}: lifetime {item.binding!r} names definition "
                f"{item.defined_at}, which is outside this function's value order"
            )
        rectangle = _Rectangle(
            binding=item.binding,
            size_bytes=item.bytes,
            first=0 if item.persistent else item.defined_at,
            last=len(order) if item.persistent else max(item.last_used_at, item.defined_at) + 1,
        )
        base = order[item.defined_at]
        for position in _positions(fn, base, level, placements, selected):
            found.setdefault((level.name, position), []).append(rectangle)
    return found


def _positions(
    fn: Function,
    base: Expr,
    level: ExplicitMemoryLevelFacts,
    placements: dict[int, Placement],
    selected: str,
) -> tuple[object, ...]:
    """Which capacity domains hold an instance of this buffer.

    A level owned per unit of the level being analysed needs the program to say
    which units hold the value. One that does not say, or a level owned per some
    other unit, leaves nothing to place it against.
    """
    if level.owner == TARGET_MEMORY_OWNER:
        return (None,)
    if level.owner != selected or level.scope != level.owner:
        raise _NotProjectable(level.name)
    placement = placements.get(id(base))
    if placement is None:
        raise _NotProjectable(level.name)
    return tuple(sorted(placement))


def _needed(rectangles: list[_Rectangle]) -> int:
    """The most bytes this domain must hold at one point in the program."""
    edges = sorted({rectangle.first for rectangle in rectangles})
    return max(
        (
            sum(
                rectangle.size_bytes
                for rectangle in rectangles
                if rectangle.first <= edge < rectangle.last
            )
            for edge in edges
        ),
        default=0,
    )


def _place(
    name: str,
    capacity: int | None,
    rectangles: list[_Rectangle],
    options: MemoryOptions,
) -> tuple[str, dict[str, int]]:
    """Give every buffer in one domain a byte range, or refuse the program.

    A domain whose whole contents fit at once needs no search: room for all of
    them is room for any arrangement of them, so they are stacked in the order
    the program defines them. Neither does one that cannot hold what is live at
    a single point, because nothing moves out of the way of something still
    being read. Every way of failing to place a buffer says which way it was,
    because those are different things to fix. The byte ranges are returned
    rather than only counted: an address a reader can act on is the point.
    """
    if not rectangles:
        return "optimal", {}
    if capacity is None or capacity <= 0:
        raise AnalysisError(
            f"the target states no usable capacity for {name!r}, so a program "
            "that places values there cannot be shown to fit"
        )
    for rectangle in rectangles:
        if rectangle.size_bytes > capacity:
            raise AnalysisError(
                f"a {name!r} buffer needs {rectangle.size_bytes} B, more than "
                f"the {capacity} B the target states for that level"
            )
    demand = _needed(rectangles)
    if demand > capacity:
        raise AnalysisError(
            f"{name!r} holds {demand} B at one point of this program, more than "
            f"the {capacity} B the target states for that level"
        )
    if sum(rectangle.size_bytes for rectangle in rectangles) <= capacity:
        stacked, cursor = {}, 0
        for rectangle in rectangles:
            stacked[rectangle.binding] = cursor
            cursor += rectangle.size_bytes
        return "optimal", stacked

    model = cp_model.CpModel()
    addresses, lives, offsets = [], [], []
    for index, rectangle in enumerate(rectangles):
        offset = model.NewIntVar(0, capacity - rectangle.size_bytes, f"o{index}")
        offsets.append(offset)
        addresses.append(
            model.NewIntervalVar(
                offset, rectangle.size_bytes, offset + rectangle.size_bytes, f"a{index}"
            )
        )
        lives.append(
            model.NewIntervalVar(
                rectangle.first, rectangle.last - rectangle.first, rectangle.last, f"l{index}"
            )
        )
    model.AddNoOverlap2D(addresses, lives)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = options.workers
    solver.parameters.random_seed = options.random_seed
    solver.parameters.max_time_in_seconds = options.timeout_seconds
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise AnalysisError(
            f"no arrangement of this program's {name!r} buffers fits the "
            f"{capacity} B the target states for that level"
        )
    if status == cp_model.MODEL_INVALID:
        raise AnalysisError(f"the {name!r} placement is not a valid solver problem")
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        raise AnalysisError(
            f"the {name!r} placement did not settle within its time limit"
        )
    placed = {
        rectangle.binding: solver.Value(offset) for rectangle, offset in zip(rectangles, offsets)
    }
    return ("optimal" if status == cp_model.OPTIMAL else "feasible"), placed


@dataclass(frozen=True)
class _Allocation:
    """One function's settled addresses, and what settling them took."""

    metadata: AllocationMetadata
    addresses: dict[tuple[str, str], int]
    placements: dict[int, Placement]


def _allocate(
    module: Module,
    fn: Function,
    record: MemoryMetadata,
    facts: MemoryHierarchyFacts,
    target: Target,
    options: MemoryOptions,
) -> _Allocation | None:
    """Address every buffer this function keeps live, and say what that took.

    Domains holding the very same buffers are one question asked again, so each
    distinct set is decided once. A function with nothing addressable to place
    gets a settled record with nothing in it: the question was asked and there
    was nothing to decide. A function whose machine names no level to place
    anything against gets no record at all, because nothing was asked. A buffer
    several domains of one level hold has one address, because it is one buffer,
    and domains that disagree about where it is are refused here.
    """
    try:
        selected = target.get_facts(ParallelCapacityFacts).topology
        module.resolve_topology(selected)
    except (UnsupportedCapabilityError, ValueError):
        return None
    placements = _buffer_placements(module, fn, target, selected)
    order = definition_order(fn)
    try:
        domains = _domains(fn, record, facts, placements, order, selected)
    except _NotProjectable:
        return None
    status = "optimal"
    addresses: dict[tuple[str, str], int] = {}
    stated: set[tuple[str, tuple[str, ...]]] = set()
    for (name, _position), rectangles in domains.items():
        signature = (name, tuple(sorted(item.binding for item in rectangles)))
        if signature in stated:
            continue
        stated.add(signature)
        level = facts.explicit(name)
        if level is None:
            continue
        settled, placed = _place(name, level.capacity_bytes, rectangles, options)
        if settled == "feasible":
            status = "feasible"
        for binding, offset in placed.items():
            held = addresses.setdefault((name, binding), offset)
            if held != offset:
                raise AnalysisError(
                    f"function {fn.name!r}: {binding!r} is placed at byte {held} "
                    f"in one {name!r} domain and at byte {offset} in another, so "
                    "it has no single address to state"
                )
    return _Allocation(AllocationMetadata(solver_status=status), addresses, placements)


def _address_buffers(
    fn: Function,
    record: MemoryMetadata,
    allocation: _Allocation,
    facts: MemoryHierarchyFacts,
    ctx,
    *,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> None:
    """Give each value the buffer its bytes are in, and where in it they start.

    A level whose capacity the target states was placed, and each value's leaves
    tile the allocation its lifetime was sized by. A level held per unit of work
    is not placed, and a value there gets a buffer of its own with nothing to
    search for: two registers are two buffers however their offsets read. A
    value living in another's bytes names that whole allocation rather than a
    range of its own, and a tuple names the buffers it was built from. Values
    are numbered as the program defines them, so it numbers them so twice.
    """
    order = definition_order(fn)
    labels = _unique_labels(order)
    carried = _carried_origins(fn)
    sizes = {(item.level, item.binding): item.bytes for item in record.lifetimes}
    seen: set[int] = set()
    walk = [
        expr
        for expr in (*fn.params, *(postorder(fn.body) if fn.body is not None else ()))
        if id(expr) not in seen and not seen.add(id(expr))
    ]

    numbers: dict[int, dict[str, int]] = {}
    minted = 0
    for expr in walk:
        if _is_view(expr) or id(expr) in carried or _parts_of(expr) is not None:
            continue
        for level in bytes_by_storage(expr.type):
            named = labels.get(id(expr))
            placed = (level, named) in allocation.addresses if named else False
            if not placed and level in _ALLOCATED_LEVELS:
                continue
            numbers.setdefault(id(expr), {}).setdefault(level, minted)
            minted += 1

    stated: dict[int, tuple[BufferRef, ...]] = {}
    for expr in walk:
        refs = _refs_of(
            expr,
            ctx=ctx,
            stated=stated,
            numbers=numbers,
            labels=labels,
            carried=carried,
            addresses=allocation.addresses,
            sizes=sizes,
            facts=facts,
            topology_levels=topology_levels,
            topologies=topologies,
        )
        if refs is None:
            continue
        stated[id(expr)] = refs
        if refs:
            attach(expr, BufferAllocationMetadata(fields=refs))


def _parts_of(expr: Expr) -> "tuple[Expr, ...] | None":
    """The values a value is made of, when it holds nothing of its own.

    A tuple names the values it was built from and a loop names what it yielded;
    neither allocates, so each of their fields is the buffer of the value behind
    it. Everything else answers for itself.
    """
    if isinstance(expr, Tuple):
        return expr.elements
    if isinstance(expr, GridRegionExpr):
        return expr.yield_values
    return None


def _renames(expr: Expr, carried: dict[int, Expr]) -> Expr:
    """The value whose bytes this one is in, one step along.

    One step and not all of them: a value that renames a field names which one,
    and that answer belongs to the step that made it. Following the chain to its
    end first loses every field named on the way.
    """
    held = carried.get(id(expr))
    if held is not None:
        return held
    alias = get_metadata(expr, BufferAliasMetadata)
    if alias is None or alias.kind not in ("forward", "update") or not alias.aliased_operands:
        return expr
    reached = expr.args[alias.aliased_operands[0]]
    return expr if reached is expr else reached


def _refs_of(
    expr: Expr,
    *,
    ctx,
    stated: dict[int, tuple[BufferRef, ...]],
    numbers: dict[int, dict[str, int]],
    labels: dict[int, str],
    carried: dict[int, Expr],
    addresses: dict[tuple[str, str], int],
    sizes: dict[tuple[str, str], int],
    facts: MemoryHierarchyFacts,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> tuple[BufferRef, ...] | None:
    """Where one value's leaves are, or None when that cannot be said."""
    parts = _parts_of(expr)
    if parts is not None:
        held = [stated.get(id(item)) for item in parts]
        return None if any(item is None for item in held) else tuple(
            ref for item in held for ref in item
        )
    owner = _renames(expr, carried)
    own = owner is expr
    if not own:
        held = stated.get(id(owner))
        if held is None:
            return None
        return _renamed(
            expr, held, _span_of(expr, ctx), facts, topology_levels, topologies
        )
    minted = numbers.get(id(owner))
    if minted is None:
        return None
    named = labels.get(id(owner))
    cursors: dict[str, int] = {}
    refs: list[BufferRef] = []
    for leaf in tensor_types(expr.type):
        if leaf.storage is StorageKind.UMAT:
            return None
        level = str(leaf.storage)
        number = minted.get(level)
        if number is None:
            return None
        held = _at_owner(leaf, level, facts, topology_levels, topologies)
        base = addresses.get((level, named), 0) if named else 0
        whole = sizes.get((level, named)) if named else None
        refs.append(
            BufferRef(
                buffer_id=number,
                level=level,
                offset=base + (cursors.get(level, 0) if own else 0),
                size=tensor_bytes(held) if own or whole is None else whole,
                shape=tuple(held.shape),
                layout=held.layout,
            )
        )
        if own:
            cursors[level] = cursors.get(level, 0) + tensor_bytes(held)
    return tuple(refs)


def _span_of(expr: Expr, ctx) -> "int | None":
    """Which field of the value it renames this one took.

    An operation that takes one field of several names it in the link that
    forwards it, and that is what decides which buffer the field is in. The
    field is named as a field rather than as a byte, so a tuple whose earlier
    fields have no static size is still placeable. One that renames the whole of
    what it reads names nothing, and starts at the front.
    """
    if ctx is None or not isinstance(expr, Call):
        return None
    handler = access_relation_registry.lookup(type(expr.target))
    if handler is None:
        return None
    try:
        relations = handler(expr, ctx)
    except (AnalysisError, NotImplementedError, ValueError):
        return None
    named = {
        link.input_field
        for output in relations.outputs
        for link in (output.storage.links if output.storage else ())
        if link.input_field is not None
    }
    return named.pop() if len(named) == 1 else None


def _renamed(
    expr: Expr,
    owner: tuple[BufferRef, ...],
    start: "int | None",
    facts: MemoryHierarchyFacts,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> tuple[BufferRef, ...] | None:
    """One value's own coordinates over the buffers another value owns.

    A value that renames all of another has its leaves, one for one. A value
    that renames one field of several is placed by the field its own operation
    named, because which field it took is what decides which buffer it is in and
    naming the first would be a guess that reads like an address.
    """
    leaves = tensor_types(expr.type)
    if len(leaves) != len(owner):
        owner = _from_field(owner, start, len(leaves))
        if owner is None:
            return None
    refs = []
    for leaf, ref in zip(leaves, owner):
        if leaf.storage is StorageKind.UMAT:
            return None
        level = str(leaf.storage)
        if level != ref.level:
            return None
        held = _at_owner(leaf, level, facts, topology_levels, topologies)
        refs.append(
            BufferRef(
                buffer_id=ref.buffer_id,
                level=level,
                offset=ref.offset,
                size=ref.size,
                shape=tuple(held.shape),
                layout=held.layout,
            )
        )
    return tuple(refs)


def _from_field(
    owner: tuple[BufferRef, ...], field: "int | None", wanted: int
) -> "tuple[BufferRef, ...] | None":
    """The run of a value's buffers that begins at the field it named."""
    if field is None or not 0 <= field <= len(owner) - wanted:
        return None
    return owner[field : field + wanted]


def _at_owner(
    leaf: Type,
    level: str,
    facts: MemoryHierarchyFacts,
    topology_levels: tuple[str, ...],
    topologies: tuple[Topology, ...],
) -> Type:
    """One leaf as the unit owning its level holds it."""
    declared = facts.explicit(level)
    if declared is None:
        return leaf
    return _type_at_owner(
        leaf,
        owner=declared.owner,
        topology_levels=topology_levels,
        topologies=topologies,
    )


def _buffer_placements(
    module: Module, fn: Function, target: Target, selected: str
) -> dict[int, Placement]:
    """Where every value that can own a buffer lives, parameters included.

    A value with no position of its own is left out rather than refused here. A
    level the target owns needs none, and one owned per position asks for it
    when it comes to place something there. The whole-program reading that
    performance requires refuses all of a program it cannot place every value
    of, which is a stricter question than this one: a value whose own type names
    its positions can be placed whatever a value elsewhere does, so each is
    asked again on its own rather than left out with a refused neighbour.
    """
    try:
        resolved = dict(
            _call_placements(module, fn, selected)
        )
    except AnalysisError:
        resolved = {}
    topology = module.resolve_topology(selected)
    for expr in (*fn.params, *postorder(fn.body)):
        if id(expr) in resolved:
            continue
        try:
            resolved[id(expr)] = _result_placement(expr.type, topology)
        except AnalysisError:
            continue
    return resolved


def _record_traffic(
    fn: Function,
    scope: FunctionScope,
    level: str | None,
    topologies: tuple[Topology, ...],
    placements: dict[int, Placement] | None = None,
    participant: int = 0,
) -> None:
    """Give every occurrence the bytes its own relations say it moves.

    A participant that does not run one does none of its movement, a share of
    zero rather than a refusal, and one whose movement cannot be stated is left
    without a record instead of an empty one. The function's own total is the
    same bytes counted as often as the loops repeat them, stated only when every
    occurrence that owed an answer gave one: a total missing a part of itself
    reads as a smaller program rather than as an incomplete count.
    """
    plan = build_buffer_plan(fn, level)
    whole = CostContext(scope=scope)
    unit = CostContext(scope=scope, level=level, topologies=topologies)
    trips = enclosing_trips(fn.body)
    totals: dict[str, list[int]] = {}
    shares: dict[str, list[int]] = {}
    complete = True
    for expr in postorder(fn.body) if fn.body is not None else ():
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        handler = access_relation_registry.lookup(type(expr.target))
        if handler is None:
            if _owes_an_answer(expr):
                complete = False
            continue
        held = (placements or {}).get(id(expr))
        try:
            moved = lower_traffic(
                expr,
                handler(expr, whole),
                handler(expr, unit),
                plan,
                whole,
                unit,
                participant=participant,
                runs=held is None or participant in held,
                umat_level=_UMAT_LEVEL,
            )
        except (AnalysisError, NotImplementedError, isl.Error):
            complete = False
            continue
        attach(expr, moved)
        repeats = trips.get(id(expr), 1)
        _add_traffic(totals, moved.whole, repeats)
        _add_traffic(shares, moved.per_unit, repeats)
    if complete:
        attach(fn, TrafficMetadata(whole=_totalled(totals), per_unit=_totalled(shares)))


def _owes_an_answer(expr: Expr) -> bool:
    """Whether an occurrence with no relation was still going to move bytes.

    An Op that states no accesses can still be one that moves something, and
    counting a function's bytes without it gives a total that is short by
    however much it moved. What says whether it moved anything is the cost
    already recorded for it; an occurrence with no cost record has not been
    shown to move nothing either.
    """
    cost = get_metadata(expr, ComputeCostMetadata)
    return cost is None or bool(cost.traffic)


def _add_traffic(
    into: dict[str, list[int]],
    stated: tuple[tuple[str, TrafficBytes], ...],
    repeats: int,
) -> None:
    """Add one occurrence's bytes to a function's, as often as it happens."""
    for level, moved in stated:
        running = into.setdefault(level, [0, 0])
        running[0] += moved.read * repeats
        running[1] += moved.write * repeats


def _totalled(found: dict[str, list[int]]) -> tuple[tuple[str, TrafficBytes], ...]:
    """One function's totals per level, in a stable order."""
    return tuple(
        (level, TrafficBytes(read, write)) for level, (read, write) in sorted(found.items())
    )


def analyze_memory(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach one memory record to every Function reachable from *function*."""
    facts = target.get_facts(MemoryHierarchyFacts)
    settings = options if isinstance(options, MemoryOptions) else MemoryOptions()
    topologies = module.effective_topologies()
    for fn in reachable_functions(function):
        try:
            residencies, length = _residencies(
                fn,
                facts=facts,
                topology_levels=target.topology_levels,
                topologies=topologies,
            )
        except ValueError as error:
            raise AnalysisError(str(error)) from None
        peaks = _peaks(residencies, length)
        persistent: dict[str, int] = {}
        for item in residencies:
            if item.persistent:
                persistent[item.level] = persistent.get(item.level, 0) + item.bytes
        loop_values = {
            id(expr): expr
            for expr in postorder(fn.body)
            if isinstance(expr, GridRegionExpr)
        }
        loop_records: list[tuple[GridRegionExpr, LoopFootprintMetadata]] = []
        for loop_id, reading in loop_footprints(module, fn).items():
            valid = tuple(
                item for item in reading.buffers if item.device_bytes >= item.bytes
            )
            loop_records.append(
                (
                    loop_values[loop_id],
                    LoopFootprintMetadata(
                        footprints=tuple(
                            BufferFootprint(
                                buffer=item.buffer,
                                level=item.level,
                                bytes=item.bytes,
                                device_bytes=item.device_bytes,
                                repeated_bytes=item.repeated_bytes,
                            )
                            for item in valid
                        ),
                        known=reading.known and len(valid) == len(reading.buffers),
                    ),
                )
            )
        record = MemoryMetadata(
            footprint=_explicit_footprint(fn, residencies, peaks, persistent, facts),
            traffic=_function_traffic(fn),
            lifetimes=tuple(
                ValueLifetime(
                    binding=item.binding,
                    level=item.level,
                    bytes=item.bytes,
                    defined_at=item.defined_at,
                    last_used_at=item.last_used_at,
                    persistent=item.persistent,
                )
                for item in residencies
            ),
            advisories=_advisories(facts, peaks, tuple(loop_records)),
        )
        allocation = _allocate(module, fn, record, facts, target, settings)
        attach(
            fn,
            replace(
                record,
                allocation=None if allocation is None else allocation.metadata,
            ),
        )
        if allocation is not None:
            scope = FunctionScope(module, fn)
            _address_buffers(
                fn,
                record,
                allocation,
                facts,
                CostContext(scope=scope),
                topology_levels=target.topology_levels,
                topologies=topologies,
            )
            _record_traffic(fn, scope, level, topologies, allocation.placements)
        for loop, footprint in loop_records:
            attach(loop, footprint)


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
    "cache_pressure",
    "definition_order",
]
