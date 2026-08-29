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


def _lifetimes(values: list[Expr], facts: MemoryHierarchyFacts) -> tuple[ValueLifetime, ...]:
    index_by_id = {id(expr): index for index, expr in enumerate(values)}
    last_by_id = dict(index_by_id)
    for index, consumer in enumerate(values):
        for operand in expr_children(consumer):
            if id(operand) in last_by_id and index > last_by_id[id(operand)]:
                last_by_id[id(operand)] = index
    result: list[ValueLifetime] = []
    for index, expr in enumerate(values):
        for level, amount in bytes_by_storage(expr.type).items():
            if facts.explicit(level) is None:
                continue
            result.append(
                ValueLifetime(
                    binding=(
                        getattr(expr, "name", None) or binding_name(expr) or f"<value {index}>"
                    ),
                    level=level,
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
    totals: dict[str, TrafficBytes] = field(default_factory=dict)
    shares: dict[str, TrafficBytes] = field(default_factory=dict)
    values: list[Expr] = field(default_factory=list)


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic while collecting lifetime order and loop footprints."""

    def visit_GridRegionExpr(self, expr: GridRegionExpr, ctx: MemoryContext) -> None:
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
    """Attach traffic and per-loop footprints from the shared Scope tree."""
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
    MemoryVisitor().visit(function.body, memory_context)
    attach(
        function,
        TrafficMetadata(
            whole=tuple(sorted(memory_context.totals.items())),
            per_unit=tuple(sorted(memory_context.shares.items())),
        ),
    )
    lifetimes = _lifetimes(memory_context.values, facts)
    topology_size = 1
    if topologies and isinstance(topologies[0].size, int):
        topology_size = max(1, topologies[0].size)
    levels_list: list[LevelFootprint] = []
    for name in sorted({item.level for item in lifetimes} | set(memory_context.shares)):
        declared = facts.explicit(name)
        divisor = topology_size if declared is not None and declared.owner != "target" else 1
        rows = [item for item in lifetimes if item.level == name]
        peak = 0
        for point in range(len(lifetimes) + 1):
            peak = max(
                peak,
                sum(
                    item.bytes // divisor
                    for item in rows
                    if item.defined_at <= point <= item.last_used_at
                ),
            )
        levels_list.append(
            LevelFootprint(
                level=name,
                peak_bytes=peak,
                persistent_bytes=sum(item.bytes // divisor for item in rows if item.persistent),
                capacity_bytes=declared.capacity_bytes if declared is not None else None,
            )
        )
    levels = tuple(levels_list)
    for level in levels:
        if level.capacity_bytes is not None and level.peak_bytes > level.capacity_bytes:
            raise AnalysisError(
                f"{level.level!r} holds {level.peak_bytes} B at one point of this program, "
                f"more than the {level.capacity_bytes} B the target states for that level"
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
