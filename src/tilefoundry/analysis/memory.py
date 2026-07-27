"""What a function needs from the memory hierarchy, and whether it fits.

The lifetimes and peaks are read from the authored program; whether a peak fits
is asked against the target's levels. The two are kept apart in the record so a
caller can see which of the numbers would survive a change of hardware.

Overflow is not one verdict. A level a program places values in by name either
has room or the program is invalid, so exceeding it fails the analysis. A cache
the program never allocated in cannot be overflowed in that sense -- a working
set larger than the cache runs, just slower -- so that finding is reported as an
advisory and nothing more.
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
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.cuda.target import CudaTarget
from tilefoundry.target.facts import TARGET_FACTS

from .errors import AnalysisError
from .facts import ImplicitMemoryLevelFacts, MemoryHierarchyFacts
from .metadata import (
    ComputeCostMetadata,
    LevelFootprint,
    MemoryMetadata,
    TrafficBytes,
    ValueLifetime,
)
from .registry import register_analysis
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


def _is_view(expr: Expr) -> bool:
    """Whether *expr* aliases its operand rather than allocating.

    A reshape or a transpose describes the same bytes differently. Charging it
    for its result would count one buffer twice.
    """
    return isinstance(expr, Call) and isinstance(expr.target, (Reshape, Transpose))


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


def _residencies(fn: Function) -> tuple[tuple[_Residency, ...], int]:
    """Every allocation resident in *fn*, with the length of its value order.

    Parameters are part of the order and start at its beginning: the function
    did not produce them, so they are already resident when it is entered. A
    parameter declared constant is a weight, and a weight is never reclaimable
    -- it stays resident past its last reader, for the whole function.
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

    result: list[_Residency] = []
    for expr in order:
        if _is_view(expr):
            continue
        persistent = isinstance(expr, Var) and expr.is_const
        for level, amount in bytes_by_storage(expr.type).items():
            result.append(
                _Residency(
                    binding=_label(expr, position[id(expr)]),
                    level=level,
                    bytes=amount,
                    defined_at=position[id(expr)],
                    last_used_at=(
                        len(order) - 1 if persistent else last_use[id(expr)]
                    ),
                    persistent=persistent,
                )
            )
    return tuple(result), len(order)


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
    """The traffic the compute-cost records on *fn*'s calls add up to."""
    totals: dict[str, TrafficBytes] = {}
    for expr in postorder(fn.body):
        record = get_metadata(expr, ComputeCostMetadata)
        if record is None:
            continue
        for level, value in record.traffic:
            current = totals.get(level, TrafficBytes())
            totals[level] = TrafficBytes(
                current.read_bytes + value.read_bytes,
                current.write_bytes + value.write_bytes,
            )
    return tuple(sorted(totals.items()))


def _explicit_footprint(
    fn: Function, peaks: dict[str, int], persistent: dict[str, int],
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
        if row.exceeds_capacity:
            raise AnalysisError(
                f"function {fn.name!r}: {level} needs {row.peak_bytes} B at its "
                f"peak, which exceeds the {row.capacity_bytes} B the target "
                f"states for that level"
            )
        rows.append(row)
    return tuple(rows)


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


def _residency_advisory(
    level: ImplicitMemoryLevelFacts,
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
) -> str | None:
    """Whether *level* is too small for the working set it fronts.

    The comparison is made only when the cache and the level it backs are stated
    per the same topology scope. A per-SM capacity set against a whole-device
    footprint would exceed it for almost any program, which reads as a finding
    while saying nothing: the per-SM share of that footprint is not known here.
    """
    backing_name = facts.backing_level(level.name)
    backing = facts.explicit(backing_name)
    if backing is None or backing.scope != level.scope:
        return None
    working_set = peaks.get(backing_name, 0)
    capacity = _implicit_capacity(level.name, facts, peaks)
    if not working_set or capacity is None or working_set <= capacity:
        return None
    return (
        f"{level.name} holds {capacity} B per {level.scope}, so the "
        f"{backing_name} working set of {working_set} B will not stay resident"
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
    facts: MemoryHierarchyFacts, peaks: dict[str, int]
) -> tuple[str, ...]:
    """Findings about the caches, none of which invalidate the program."""
    notes: list[str] = []
    for level in facts.implicit_levels:
        notes.extend(
            note
            for note in (
                _division_advisory(level, facts, peaks),
                _residency_advisory(level, facts, peaks),
            )
            if note is not None
        )
    return tuple(notes)


def analyze_memory(
    module: Module,
    function: Function,
    target: object,
    options: object | None = None,
) -> None:
    """Attach one memory record to every Function reachable from *function*."""
    facts = TARGET_FACTS.project(target, MemoryHierarchyFacts)
    for fn in reachable_functions(function):
        residencies, length = _residencies(fn)
        peaks = _peaks(residencies, length)
        persistent: dict[str, int] = {}
        for item in residencies:
            if item.persistent:
                persistent[item.level] = persistent.get(item.level, 0) + item.bytes
        attach(
            fn,
            MemoryMetadata(
                footprint=_explicit_footprint(fn, peaks, persistent, facts),
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
                advisories=_advisories(facts, peaks),
            ),
        )


for _target_type in (CudaTarget, AmxTarget):
    register_analysis(
        _target_type,
        SELECTOR,
        requires=("compute-cost",),
        produces=(MemoryMetadata,),
    )(analyze_memory)


__all__ = ["SELECTOR", "analyze_memory"]
