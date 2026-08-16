"""Measure function residency against the target memory hierarchy.

Lifetime order comes from authored IR and residency is projected to each
explicit level's owner. A single value exceeding a level is an error because no
schedule can place it. Peak overflow and cache working-set overflow are
advisories because ordering can change the former and the latter affects speed.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Var,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import Type, local_type_of
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import Target

from .errors import AnalysisError
from .facts import (
    TARGET_MEMORY_OWNER,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
)
from .footprint import loop_footprints
from .metadata import (
    BufferFootprint,
    ComputeCostMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    MemoryMetadata,
    TrafficBytes,
    ValueLifetime,
)
from .walk import (
    attach,
    bytes_by_storage,
    children,
    postorder,
    reachable_functions,
)

SELECTOR = "memory"


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
    """Whether *expr* aliases its operand rather than allocating.

    Reshape and Slice re-index the same elements. Other operations produce
    values at addresses of their own; buffer aliasing may optimize those later.
    """
    return isinstance(expr, Call) and isinstance(expr.target, (Reshape, Slice))


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
    order: list[Expr] = [
        *fn.params,
        *(expr for expr in postorder(fn.body) if isinstance(expr, (Call, Constant))),
    ]
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


def analyze_memory(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach one memory record to every Function reachable from *function*."""
    facts = target.get_facts(MemoryHierarchyFacts)
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
        attach(
            fn,
            MemoryMetadata(
                footprint=_explicit_footprint(
                    fn, residencies, peaks, persistent, facts
                ),
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
            ),
        )
        for loop, record in loop_records:
            attach(loop, record)


__all__ = ["SELECTOR", "analyze_memory", "cache_pressure"]
