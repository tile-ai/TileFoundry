"""Memory-family projection from shared Scope and Access records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Var,
    VerifyError,
    binding_name,
    describe_expr,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import TensorType, TupleType, Type, bytes_by_storage
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprVisitor, expr_children
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    leaves_of,
    projected,
    reached_elements,
    reached_leaves,
    relations_of,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import MemoryHierarchyFacts
from .metadata import (
    AllocationMetadata,
    LevelFootprint,
    LoopFootprintMetadata,
    MemoryMetadata,
    TrafficBytes,
    TrafficMetadata,
    ValueLifetime,
)
from .scope import Scope
from .visitor import AnalyzeContext

SELECTOR = "memory"
_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


@dataclass(frozen=True)
class MemoryOptions:
    """How long the placement may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


def _reached_bytes(
    boundaries: tuple[tuple[Type, object], ...], umat_level: str | None
) -> tuple[int, dict[str, int]] | None:
    """What one operand's boundaries reach, in bytes and per level.

    A structured operand is indexed by leaf and its leaves need not be the same
    width or live at the same level, so which ones a boundary reaches decides
    both numbers: charging the first for the one that was taken is a wrong
    number at the right size. A single leaf is counted in its own elements
    instead. A leaf nobody materialised is part of what moved and part of no
    level's traffic unless the caller says where this occurrence puts it.
    """
    total = 0
    by_level: dict[str, int] = {}
    for held, pattern in boundaries:
        leaves = leaves_of(held)
        if not leaves:
            return None
        if len(leaves) == 1:
            taken = {0: _bytes_for(leaves[0], reached_elements(pattern))}
        else:
            reached = reached_leaves(pattern, len(leaves))
            if reached is None:
                return None
            taken = {index: static_bytes(leaves[index]) for index in sorted(reached)}
        for index, size in taken.items():
            if size is None:
                return None
            total += size
            leaf = leaves[index]
            level = umat_level if leaf.storage is StorageKind.UMAT else str(leaf.storage)
            if level is not None:
                by_level[level] = by_level.get(level, 0) + size
    return total, by_level


def _bytes_for(held: Type, elements: int | None) -> int | None:
    """The bytes *elements* of *held* occupy, or ``None`` when unanswerable.

    Counted from how wide one element is, rounded up: a packed dtype has no
    whole number of bytes per element, so a share of the whole would round one
    bool to nothing and nine of them to one byte instead of two. A value whose
    element width nobody states is not one this can answer for.
    """
    if elements is None or not isinstance(held, TensorType):
        return None
    bits = getattr(held.dtype, "bit_width", None)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        return None
    return -(-elements * bits // 8)


def _movement(
    call: Call,
    cost: Cost,
    ctx: CostContext,
    types: tuple[Type, ...],
    stated_relations: AccessRelations | None = None,
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and the levels those bytes are at.

    A Type says how big a value is, not how much of it this occurrence touches,
    so the amount is what that boundary's relation reaches in this context's
    window -- one handler answering for the whole program and for one unit. The
    direction stays the cost's answer and the level is where the reached leaf
    lives, so one leaf of two owes its own bytes at its own level. A Function has
    no boundaries of its own, and every other target states its coordinates or
    is refused, as is a boundary nothing can charge in bytes.
    """
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe_expr(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    traffic = cost.traffic
    if isinstance(call.target, Function):
        charged = [
            bytes_by_storage(
                type_,
                umat_level=_UMAT_CONSUMPTION_LEVEL if index < len(call.args) else None,
            )
            for index, type_ in enumerate(types)
        ]
    else:
        if access_relation_registry.lookup(type(call.target)) is None:
            raise AnalysisError(
                f"{describe_expr(call)}: states no access relations, so nothing here "
                "says what it moves"
            )
        stated_relations = (
            stated_relations if stated_relations is not None else relations_of(call, ctx)
        )
        local_relations = projected(stated_relations, call, ctx)
        result = ctx.local_type_of(call)
        fields = result.fields if isinstance(result, TupleType) else (result,)
        if len(fields) > len(local_relations.outputs):
            raise AnalysisError(
                f"{describe_expr(call)}: states {len(local_relations.outputs)} output "
                f"boundaries for a result of {len(fields)} fields"
            )
        amounts, charged = [], []
        for index, moved in enumerate(cost.traffic):
            if index == len(call.args):
                asked = tuple(
                    (field_, local_relations.outputs[position].pattern)
                    for position, field_ in enumerate(fields)
                )
                level = None
            else:
                asked = ((types[index], local_relations.inputs[index].pattern),)
                level = _UMAT_CONSUMPTION_LEVEL
            answer = _reached_bytes(asked, level)
            if answer is None:
                raise AnalysisError(
                    f"{describe_expr(call)}: boundary {index} reaches coordinates "
                    "nothing here can charge in bytes"
                )
            moving, by_level = answer
            amounts.append(TrafficBytes(moving if moved.read else 0, moving if moved.write else 0))
            charged.append(by_level)
        traffic = tuple(amounts)
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    for moved, split in zip(traffic, charged):
        for level, value in split.items():
            if moved.read:
                reads[level] = reads.get(level, 0) + value
            if moved.write:
                writes[level] = writes.get(level, 0) + value
    moving = any(item.read or item.write for item in traffic)
    levels = (
        tuple(
            (level, TrafficBytes(reads.get(level, 0), writes.get(level, 0)))
            for level in sorted(set(reads) | set(writes))
        )
        if moving
        else ()
    )
    return levels, traffic


def call_traffic(
    expr: Call,
    whole: CostContext,
    local: CostContext,
    stated_relations: AccessRelations | None = None,
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    The Type of the leaf it reached names the level those bytes are charged at,
    and an allocation does not correct either answer.
    """
    try:
        whole_cost = CostEvaluator().visit(expr, whole)
        local_cost = CostEvaluator().visit(expr, local)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    asked = []
    for ctx, cost in ((whole, whole_cost), (local, local_cost)):
        types = (
            *(ctx.local_type_of(arg) for arg in expr.args),
            ctx.local_type_of(expr),
        )
        asked.append(_movement(expr, cost, ctx, types, stated_relations))
    (whole_levels, operands), (unit_levels, _unit_operands) = asked
    return TrafficMetadata(whole=whole_levels, per_unit=unit_levels, operands=operands)


def add_traffic(
    whole: dict[str, TrafficBytes],
    per_unit: dict[str, TrafficBytes],
    record: TrafficMetadata,
    trips: int,
) -> None:
    """Add one occurrence's bytes to a function's, as often as it happens."""
    for into, stated in ((whole, record.whole), (per_unit, record.per_unit)):
        for level, moved in stated:
            running = into.get(level, TrafficBytes())
            into[level] = TrafficBytes(
                running.read + moved.read * trips,
                running.write + moved.write * trips,
            )


@dataclass(frozen=True)
class _Residency:
    """One lexical scope's own residencies and where its loops sit among them.

    ``points`` are the positions occupancy is asked about: this scope's own
    value positions plus one past them, so a value whose last reader is a loop
    written after every other value still has a point that holds it. ``nested``
    pairs each child scope with the point of this scope it occupies.
    """

    rows: tuple[ValueLifetime, ...]
    points: tuple[int, ...]
    nested: tuple[tuple[int, "_Residency"], ...]


def _read_below(scope: Scope, values: dict[int, list[Expr]]) -> set[int]:
    """Ids of every value read at or below *scope*, its loop's operands included.

    A loop reads its own init and yield operands, so an enclosing value the body
    never touches is still resident for as long as the loop carries it.
    """
    reached = {id(operand) for expr in values[id(scope)] for operand in expr_children(expr)}
    if isinstance(scope.owner, GridRegionExpr):
        reached |= {id(operand) for operand in expr_children(scope.owner)}
    for child in scope.children:
        reached |= _read_below(child, values)
    return reached


def _rows(
    scope: Scope,
    values: dict[int, list[Expr]],
    sites: dict[int, int],
    facts: MemoryHierarchyFacts,
    local: CostContext,
    base: int,
) -> tuple[ValueLifetime, ...]:
    """This scope's residencies, positioned from *base* in its own order.

    How much one value holds comes from the same projection the traffic family
    reads, so both halves of one report answer for the same unit. Taking the
    whole tensor here and dividing by a per-level constant outside was a second
    account of the same quantity, and the two disagreed by the ratio between a
    value's own shard factor and the topology's declared unit count. A value a
    nested scope reads is resident until the point that scope sits at.
    """
    own = values[id(scope)]
    last_by_id = {id(expr): index for index, expr in enumerate(own)}
    for index, consumer in enumerate(own):
        for operand in expr_children(consumer):
            if id(operand) in last_by_id and index > last_by_id[id(operand)]:
                last_by_id[id(operand)] = index
    for child in scope.children:
        site = sites[id(child)]
        for key in _read_below(child, values):
            if key in last_by_id and site > last_by_id[key]:
                last_by_id[key] = site
    result: list[ValueLifetime] = []
    for index, expr in enumerate(own):
        for level, amount in bytes_by_storage(local.local_type_of(expr)).items():
            if facts.explicit(level) is None:
                continue
            position = base + index
            result.append(
                ValueLifetime(
                    binding=(
                        getattr(expr, "name", None) or binding_name(expr) or f"<value {position}>"
                    ),
                    level=level,
                    bytes=amount,
                    defined_at=position,
                    last_used_at=base
                    + (
                        max(len(own) - 1, last_by_id[id(expr)])
                        if isinstance(expr, Var)
                        else last_by_id[id(expr)]
                    ),
                    persistent=isinstance(expr, Var),
                )
            )
    return tuple(result)


def _residency(
    scope: Scope,
    values: dict[int, list[Expr]],
    sites: dict[int, int],
    facts: MemoryHierarchyFacts,
    local: CostContext,
    base: int,
) -> tuple[_Residency, int]:
    """Build the residency tree, numbering scopes depth-first from *base*.

    Positions stay unique across the report: a scope owns ``base`` through
    ``base + len(own)``, the last of them reserved as the point its loops sit
    at, and its children are numbered after that.
    """
    own = values[id(scope)]
    rows = _rows(scope, values, sites, facts, local, base)
    cursor = base + len(own) + 1
    nested: list[tuple[int, _Residency]] = []
    for child in scope.children:
        below, cursor = _residency(child, values, sites, facts, local, cursor)
        nested.append((base + sites[id(child)], below))
    return _Residency(rows, tuple(range(base, base + len(own) + 1)), tuple(nested)), cursor


def _lifetimes(record: _Residency) -> tuple[ValueLifetime, ...]:
    """Every residency below *record*, outer scope before the loops it holds."""
    return (
        *record.rows,
        *(item for _site, below in record.nested for item in _lifetimes(below)),
    )


def _peak(record: _Residency, level: str, carried: int) -> int:
    """One level's worst simultaneous residency at or below *record*.

    A scope's own worst point, on top of what every enclosing scope holds at the
    point this one sits. Sequential siblings are separate branches and take the
    greater, not the sum, and a loop body counts once rather than once per trip:
    residency is a stock, not a flow. Persistent bytes are resident at every
    point of the function, so they are added rather than read off an interval. A
    loop sits between two of its scope's positions, so a value defined at that
    position is the loop's reader and not yet resident while it runs.
    """
    rows = tuple(item for item in record.rows if item.level == level)
    resident = sum(item.bytes for item in rows if item.persistent)
    transient = tuple(item for item in rows if not item.persistent)

    def held(point: int) -> int:
        return sum(
            item.bytes for item in transient if item.defined_at <= point <= item.last_used_at
        )

    def across(site: int) -> int:
        return sum(item.bytes for item in transient if item.defined_at < site <= item.last_used_at)

    peak = carried + resident + max((held(point) for point in record.points), default=0)
    for site, below in record.nested:
        peak = max(peak, _peak(below, level, carried + resident + across(site)))
    return peak


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    whole: CostContext | None = None
    local: CostContext | None = None
    totals: dict[str, TrafficBytes] = field(default_factory=dict)
    shares: dict[str, TrafficBytes] = field(default_factory=dict)
    values: list[Expr] = field(default_factory=list)
    by_scope: dict[int, list[Expr]] = field(default_factory=dict)
    sites: dict[int, int] = field(default_factory=dict)


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic while collecting lifetime order and loop footprints."""

    def visit_GridRegionExpr(self, expr: GridRegionExpr, ctx: MemoryContext) -> None:
        """Walk one loop with its own value sequence, and record where it sits.

        A fresh list, not the same one: ``replace`` passes the field by
        reference, so a loop body used to append into its parent's sequence and
        the nesting the scope tree already records was lost before the residency
        tree could read it. The two dicts are shared for the same reason.
        """
        child = next(item for item in ctx.current.children if item.owner is expr)
        for operand in expr.init_args:
            self.visit(operand, ctx)
        inner = replace(ctx, current=child, values=[])
        ctx.by_scope[id(child)] = inner.values
        ctx.sites[id(child)] = len(ctx.values)
        self.visit(expr.body, inner)
        for operand in expr.yield_values:
            self.visit(operand, inner)
        attach(expr, child.footprint())

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: MemoryContext
    ) -> None:
        if isinstance(expr, (Call, Constant)):
            ctx.values.append(expr)
        if not isinstance(expr, Call):
            return
        recorded = id(expr) in ctx.current.accesses["narrow"]
        if ctx.whole is None or ctx.local is None:
            raise AnalysisError("memory: visitor context is missing cost contexts")
        moved = (
            call_traffic(
                expr,
                ctx.whole,
                ctx.local,
                ctx.current.stated_relations(expr, ctx.whole),
            )
            if recorded
            else TrafficMetadata()
        )
        attach(expr, moved)
        if not recorded:
            return
        repeats = 1
        cursor = ctx.current
        while cursor.parent is not None:
            if cursor.is_variant(expr):
                repeats *= max(1, cursor.trips())
            cursor = cursor.parent
        add_traffic(ctx.totals, ctx.shares, moved, repeats)


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Attach traffic and per-loop footprints from the shared Scope tree.

    The peak is nesting-aware and needs no solver: a value is live from its
    definition to its last use in its own scope, a loop body's values sit on top
    of what the enclosing scopes hold at the point the loop occupies, and a loop
    counts once rather than once per trip. What is refused is one value against
    its level's capacity, not the working set: a shared budget spent whole on one
    buffer is a program the sum would refuse and the machine can still run, and
    the wording is the one the authoring tutorial prints.
    """
    module = context.module
    level = context.level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    topologies = module.effective_topologies()
    whole = CostContext(scope=FunctionScope(module, function))
    local = CostContext(scope=FunctionScope(module, function), level=level, topologies=topologies)
    memory_context = MemoryContext(
        module=module,
        target=context.target,
        level=level,
        options=context.options,
        root=context.root,
        current=context.current,
        whole=whole,
        local=local,
        values=list(function.params),
    )
    memory_context.by_scope[id(context.current)] = memory_context.values
    MemoryVisitor().visit(function.body, memory_context)
    attach(
        function,
        TrafficMetadata(
            whole=tuple(sorted(memory_context.totals.items())),
            per_unit=tuple(sorted(memory_context.shares.items())),
        ),
    )
    residency, _end = _residency(
        context.current, memory_context.by_scope, memory_context.sites, facts, local, 0
    )
    lifetimes = _lifetimes(residency)
    levels_list: list[LevelFootprint] = []
    for name in sorted({item.level for item in lifetimes} | set(memory_context.shares)):
        declared = facts.explicit(name)
        rows = [item for item in lifetimes if item.level == name]
        levels_list.append(
            LevelFootprint(
                level=name,
                peak_bytes=_peak(residency, name, 0),
                persistent_bytes=sum(item.bytes for item in rows if item.persistent),
                capacity_bytes=declared.capacity_bytes if declared is not None else None,
            )
        )
    levels = tuple(levels_list)
    for item in lifetimes:
        declared = facts.explicit(item.level)
        capacity = None if declared is None else declared.capacity_bytes
        if capacity is not None and item.bytes > capacity:
            raise AnalysisError(
                f"function {function.name!r}: value {item.binding!r} needs "
                f"{item.bytes} B in {item.level}, which exceeds the "
                f"{capacity} B the target states for that level"
            )
    attach(
        function,
        MemoryMetadata(
            footprint=levels,
            lifetimes=lifetimes,
            allocation=AllocationMetadata(solver_status="optimal"),
        ),
    )


def cache_pressure(
    record: LoopFootprintMetadata,
    facts: MemoryHierarchyFacts,
    peaks: dict[str, int],
) -> tuple[dict[str, object], ...]:
    """Compare one scope's device footprint with same-scope implicit caches."""
    rows: list[dict[str, object]] = []
    for level in facts.implicit_levels:
        backing_name = facts.backing_level(level.name)
        backing = facts.explicit(backing_name)
        if backing is None or backing.scope != level.scope:
            continue
        accesses = tuple(item for item in record.footprints if item.level == backing_name)
        if not accesses or any(item.device_bytes < item.bytes for item in accesses):
            continue
        working_set = sum(item.device_bytes for item in accesses)
        capacity = level.capacity_bytes
        for peer, shared_bytes in facts.capacity_sharers(level.name):
            if shared_bytes is None:
                continue
            remaining = shared_bytes - peaks.get(peer, 0)
            capacity = remaining if capacity is None else min(capacity, remaining)
        status = (
            "unknown"
            if capacity is None
            else "exceeds"
            if working_set > capacity
            else "fits"
            if record.known
            else "lower-bound"
        )
        rows.append(
            {
                "cache_level": level.name,
                "backing_level": backing_name,
                "device_bytes": working_set,
                "capacity_bytes": capacity,
                "status": status,
            }
        )
    return tuple(rows)


__all__ = ["MemoryOptions", "SELECTOR", "analyze_memory", "cache_pressure"]
