"""Memory-family projection from shared Scope and Access records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Var,
    VerifyError,
    describe_expr,
    value_labels,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
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
    Breakdown,
    LoopFootprintMetadata,
    MemoryLevelFootprint,
    MemoryMetadata,
    Spread,
    TrafficBytes,
    TrafficMetadata,
    ValueLifetime,
)
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
    by_memory_level: dict[str, int] = {}
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
            memory_level = umat_level if leaf.storage is StorageKind.UMAT else str(leaf.storage)
            if memory_level is not None:
                by_memory_level[memory_level] = by_memory_level.get(memory_level, 0) + size
    return total, by_memory_level


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
                memory_level = None
            else:
                asked = ((types[index], local_relations.inputs[index].pattern),)
                memory_level = _UMAT_CONSUMPTION_LEVEL
            answer = _reached_bytes(asked, memory_level)
            if answer is None:
                raise AnalysisError(
                    f"{describe_expr(call)}: boundary {index} reaches coordinates "
                    "nothing here can charge in bytes"
                )
            moving, by_memory_level = answer
            amounts.append(TrafficBytes(moving if moved.read else 0, moving if moved.write else 0))
            charged.append(by_memory_level)
        traffic = tuple(amounts)
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    for moved, split in zip(traffic, charged):
        for memory_level, value in split.items():
            if moved.read:
                reads[memory_level] = reads.get(memory_level, 0) + value
            if moved.write:
                writes[memory_level] = writes.get(memory_level, 0) + value
    moving = any(item.read or item.write for item in traffic)
    levels = (
        tuple(
            (name, TrafficBytes(reads.get(name, 0), writes.get(name, 0)))
            for name in sorted(set(reads) | set(writes))
        )
        if moving
        else ()
    )
    return levels, traffic


def call_traffic(
    expr: Call,
    whole: CostContext,
    locals_by_unit: "dict[str, CostContext]",
    stated_relations: AccessRelations | None = None,
    asked: "str | None" = None,
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    The Type of the leaf it reached names the level those bytes are charged at,
    and an allocation does not correct either answer.
    """
    storage: dict[str, dict[str, TrafficBytes]] = {}
    crossing: dict[str, dict[str, TrafficBytes]] = {}
    operands: tuple[TrafficBytes, ...] = ()
    for key, ctx in (("", whole), *locals_by_unit.items()):
        try:
            cost = CostEvaluator().visit(expr, ctx)
        except (ValueError, VerifyError) as error:
            raise AnalysisError(str(error)) from None
        types = (
            *(ctx.local_type_of(arg) for arg in expr.args),
            ctx.local_type_of(expr),
        )
        levels, positional = _movement(expr, cost, ctx, types, stated_relations)
        for memory_level, moved in levels:
            storage.setdefault(memory_level, {})[key] = moved
        for boundary, moved in cost.sent:
            if key and _finer_than(key, boundary, locals_by_unit):
                continue
            crossing.setdefault(boundary, {})[key] = moved
        if not key:
            operands = positional
    return TrafficMetadata(
        topologies=tuple(locals_by_unit),
        storage=_shares(storage, tuple(locals_by_unit)),
        communication=_shares(crossing, tuple(locals_by_unit)),
        operands=operands,
    )


def _finer_than(unit: str, boundary: str, ordered: "dict[str, CostContext]") -> bool:
    """Whether *unit* sits inside *boundary* rather than at or around it.

    Crossing a boundary is what the units on either side of it do. A unit
    inside one did not cross it, and stating a share for it would divide a
    move nobody made.
    """
    names = tuple(ordered)
    if unit not in names or boundary not in names:
        return False
    return names.index(unit) > names.index(boundary)


_WHOLE = ""


def _shares(
    held: "dict[str, dict[str, TrafficBytes]]", topologies: tuple[str, ...]
) -> "Breakdown[TrafficBytes]":
    """One entry per level, each carrying the whole and every level's share.

    The shares run in *topologies* order, so a level a move never reached --
    a unit inside the boundary it crossed -- reads as no bytes rather than as
    a missing entry.
    """
    return Breakdown(
        tuple(
            (
                name,
                Spread(
                    shares.get(_WHOLE, TrafficBytes()),
                    tuple(shares.get(unit, TrafficBytes()) for unit in topologies),
                ),
            )
            for name, shares in sorted(held.items())
        )
    )


def add_traffic(
    whole: "dict[str, dict[str, TrafficBytes]]",
    per_unit: "dict[str, dict[str, TrafficBytes]]",
    record: TrafficMetadata,
    trips: int,
) -> None:
    """Add one occurrence's bytes to a function's, as often as it happens."""
    for into, stated in ((whole, record.storage), (per_unit, record.communication)):
        for name, spread in stated.kinds:
            for key, moved in (
                (_WHOLE, spread.total),
                *zip(record.topologies, spread.per_unit, strict=False),
            ):
                running = into.setdefault(name, {}).get(key, TrafficBytes())
                into[name][key] = TrafficBytes(
                    running.read + moved.read * trips,
                    running.write + moved.write * trips,
                )


def _lifetimes(
    values: list[Expr], facts: MemoryHierarchyFacts, local: CostContext
) -> tuple[ValueLifetime, ...]:
    """Each value's residency, its bytes taken in the analysed level's window.

    The projection is the one the traffic family reads, so both halves of a report
    answer for the same unit. Dividing the whole tensor by a per-level constant
    outside was a second account of it, and the two disagreed.
    """
    index_by_id = {id(expr): index for index, expr in enumerate(values)}
    last_by_id = dict(index_by_id)
    for index, consumer in enumerate(values):
        for operand in expr_children(consumer):
            if id(operand) in last_by_id and index > last_by_id[id(operand)]:
                last_by_id[id(operand)] = index
    result: list[ValueLifetime] = []
    labels = value_labels(values)
    for index, expr in enumerate(values):
        for memory_level, amount in bytes_by_storage(local.local_type_of(expr)).items():
            if facts.explicit(memory_level) is None:
                continue
            result.append(
                ValueLifetime(
                    binding=labels[index],
                    memory_level=memory_level,
                    bytes=amount,
                    defined_at=index,
                    last_used_at=(
                        len(values) - 1 if isinstance(expr, Var) else last_by_id[id(expr)]
                    ),
                    persistent=isinstance(expr, Var),
                )
            )
    return tuple(result)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    whole: CostContext | None = None
    local: CostContext | None = None
    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    totals: dict[str, dict[str, TrafficBytes]] = field(default_factory=dict)
    shares: dict[str, dict[str, TrafficBytes]] = field(default_factory=dict)
    values: list[Expr] = field(default_factory=list)


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic while collecting lifetime order and loop footprints."""

    def visit_LoopRegion(self, expr: LoopRegion, ctx: MemoryContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.init_args:
            self.visit(operand, ctx)
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
        if ctx.whole is None or not ctx.locals_by_unit:
            raise AnalysisError("memory: visitor context is missing cost contexts")
        moved = (
            call_traffic(
                expr,
                ctx.whole,
                ctx.locals_by_unit,
                ctx.current.stated_relations(expr, ctx.whole),
                ctx.topology_level,
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
    """Attach traffic and per-loop footprints from the shared Scope tree."""
    module = context.module
    topology_level = context.topology_level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    topologies = module.effective_topologies()
    whole = CostContext(scope=FunctionScope(module, function))
    local = CostContext(
        scope=FunctionScope(module, function),
        topology_level=topology_level,
        topologies=topologies,
    )
    units = tuple(topology.name for topology in topologies) or (
        (topology_level,) if topology_level else ()
    )
    locals_by_unit = {
        unit: CostContext(
            scope=FunctionScope(module, function),
            topology_level=unit,
            topologies=topologies,
        )
        for unit in units
    }
    memory_context = MemoryContext(
        module=module,
        target=context.target,
        topology_level=topology_level,
        options=context.options,
        root=context.root,
        current=context.current,
        whole=whole,
        local=local,
        locals_by_unit=locals_by_unit,
        values=list(function.params),
    )
    MemoryVisitor().visit(function.body, memory_context)
    attach(
        function,
        TrafficMetadata(
            topologies=tuple(locals_by_unit),
            storage=_shares(memory_context.totals, tuple(locals_by_unit)),
            communication=_shares(memory_context.shares, tuple(locals_by_unit)),
        ),
    )
    lifetimes = _lifetimes(memory_context.values, facts, local)
    levels_list: list[MemoryLevelFootprint] = []
    for name in sorted({item.memory_level for item in lifetimes} | set(memory_context.totals)):
        declared = facts.explicit(name)
        rows = [item for item in lifetimes if item.memory_level == name]
        peak = 0
        for point in range(len(lifetimes) + 1):
            peak = max(
                peak,
                sum(item.bytes for item in rows if item.defined_at <= point <= item.last_used_at),
            )
        levels_list.append(
            MemoryLevelFootprint(
                memory_level=name,
                peak_bytes=peak,
                persistent_bytes=sum(item.bytes for item in rows if item.persistent),
                capacity_bytes=declared.capacity_bytes if declared is not None else None,
            )
        )
    levels = tuple(levels_list)
    for item in lifetimes:
        declared = facts.explicit(item.memory_level)
        capacity = None if declared is None else declared.capacity_bytes
        if capacity is not None and item.bytes > capacity:
            raise AnalysisError(
                f"function {function.name!r}: value {item.binding!r} needs "
                f"{item.bytes} B in {item.memory_level}, which exceeds the "
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
    for cache in facts.implicit_levels:
        backing_name = facts.backing_level(cache.name)
        backing = facts.explicit(backing_name)
        if backing is None or backing.scope != cache.scope:
            continue
        accesses = tuple(item for item in record.footprints if item.memory_level == backing_name)
        if not accesses or any(item.device_bytes < item.bytes for item in accesses):
            continue
        working_set = sum(item.device_bytes for item in accesses)
        capacity = cache.capacity_bytes
        for peer, shared_bytes in facts.capacity_sharers(cache.name):
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
                "cache_level": cache.name,
                "backing_level": backing_name,
                "device_bytes": working_set,
                "capacity_bytes": capacity,
                "status": status,
            }
        )
    return tuple(rows)


__all__ = ["MemoryOptions", "SELECTOR", "analyze_memory", "cache_pressure"]
