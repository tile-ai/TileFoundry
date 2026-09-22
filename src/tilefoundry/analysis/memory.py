"""Memory-family projection from shared IterationScope and Access records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    VerifyError,
    describe_expr,
    value_labels,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.types import TensorType, TupleType, Type, bytes_by_storage
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprVisitor
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

from .allocation import AllocationValue, solve_allocation
from .errors import AnalysisError
from .facts import MemoryHierarchyFacts
from .liveness import Liveness, analyze_liveness
from .metadata import (
    Breakdown,
    MemoryLevelPeak,
    MemoryMetadata,
    RegionMemoryMetadata,
    Spread,
    Traffic,
    TrafficBytes,
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
) -> MemoryMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    The Type of the leaf it reached names the level those bytes are charged at,
    and an allocation does not correct either answer.
    """
    storage_whole: dict[str, TrafficBytes] = {}
    storage_per_unit: dict[str, dict[str, TrafficBytes]] = {}
    crossing_whole: dict[str, TrafficBytes] = {}
    crossing_per_unit: dict[str, dict[str, TrafficBytes]] = {}
    operands: tuple[TrafficBytes, ...] = ()
    for unit, ctx in ((None, whole), *locals_by_unit.items()):
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
            if unit is None:
                storage_whole[memory_level] = moved
            else:
                storage_per_unit.setdefault(memory_level, {})[unit] = moved
        for boundary, moved in cost.sent:
            if unit is not None and _finer_than(unit, boundary, locals_by_unit):
                continue
            if unit is None:
                crossing_whole[boundary] = moved
            else:
                crossing_per_unit.setdefault(boundary, {})[unit] = moved
        if unit is None:
            operands = positional
    return MemoryMetadata(
        topologies=tuple(locals_by_unit),
        traffic=Traffic(
            storage=_occurrence_shares(storage_whole, storage_per_unit, tuple(locals_by_unit)),
            communication=_occurrence_shares(
                crossing_whole, crossing_per_unit, tuple(locals_by_unit)
            ),
        ),
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


def _occurrence_shares(
    whole: dict[str, TrafficBytes],
    per_unit: dict[str, dict[str, TrafficBytes]],
    topologies: tuple[str, ...],
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
                    logical=whole.get(name, TrafficBytes()),
                    total=whole.get(name, TrafficBytes()),
                    per_unit=tuple(
                        per_unit.get(name, {}).get(unit, TrafficBytes()) for unit in topologies
                    ),
                ),
            )
            for name in sorted({*whole, *per_unit})
        )
    )


@dataclass
class _TrafficAccounts:
    """Function traffic kept separate in its three counting domains."""

    logical: dict[str, TrafficBytes] = field(default_factory=dict)
    total: dict[str, TrafficBytes] = field(default_factory=dict)
    per_unit: dict[str, dict[str, TrafficBytes]] = field(default_factory=dict)


def _account_shares(
    accounts: _TrafficAccounts, topologies: tuple[str, ...]
) -> Breakdown[TrafficBytes]:
    """Build a breakdown without encoding an account as a topology key."""
    names = sorted({*accounts.logical, *accounts.total, *accounts.per_unit})
    return Breakdown(
        tuple(
            (
                name,
                Spread(
                    logical=accounts.logical.get(name, TrafficBytes()),
                    total=accounts.total.get(name, TrafficBytes()),
                    per_unit=tuple(
                        accounts.per_unit.get(name, {}).get(unit, TrafficBytes())
                        for unit in topologies
                    ),
                ),
            )
            for name in names
        )
    )


def _accumulate(into: dict[str, TrafficBytes], name: str, moved: TrafficBytes, trips: int) -> None:
    running = into.get(name, TrafficBytes())
    into[name] = TrafficBytes(
        running.read + moved.read * trips,
        running.write + moved.write * trips,
    )


def add_traffic(
    storage: _TrafficAccounts,
    communication: _TrafficAccounts,
    record: MemoryMetadata,
    logical_trips: int,
    total_trips: int,
) -> None:
    """Add one occurrence to each independent Function counting domain."""
    for into, stated in (
        (storage, record.traffic.storage),
        (communication, record.traffic.communication),
    ):
        for name, spread in stated.kinds:
            _accumulate(into.logical, name, spread.logical, logical_trips)
            _accumulate(into.total, name, spread.total, total_trips)
            for unit, moved in zip(record.topologies, spread.per_unit, strict=False):
                _accumulate(into.per_unit.setdefault(name, {}), unit, moved, total_trips)


def _resident_value_ids(function: Function, liveness: Liveness) -> frozenset[int]:
    """Values whose SSA interval represents independently resident bytes."""
    result = {id(parameter) for parameter in function.params}
    for interval in liveness.intervals:
        value = interval.value
        if isinstance(value, (Call, Constant, LoopRegion)):
            result.add(id(value))
        if isinstance(value, LoopRegion):
            result.update(id(phi) for phi in value.carried_args)
    return frozenset(result)


def _project_allocation_values(
    liveness: Liveness,
    resident_ids: frozenset[int],
    parameter_ids: frozenset[int],
    facts: MemoryHierarchyFacts,
    local: CostContext,
) -> tuple[AllocationValue, ...]:
    """Project structural intervals into the analysed topology window."""
    intervals = tuple(
        interval for interval in liveness.intervals if id(interval.value) in resident_ids
    )
    result: list[AllocationValue] = []
    labels = value_labels(interval.value for interval in intervals)
    for label, interval in zip(labels, intervals, strict=True):
        expr = interval.value
        persistent = id(expr) in parameter_ids
        for memory_level, amount in bytes_by_storage(local.local_type_of(expr)).items():
            if facts.explicit(memory_level) is None:
                continue
            result.append(
                AllocationValue(
                    expr,
                    ValueLifetime(
                        binding=label,
                        memory_level=memory_level,
                        bytes=amount,
                        defined_at=interval.defined_at,
                        last_used_at=(
                            liveness.timeline_end if persistent else interval.last_used_at
                        ),
                        persistent=persistent,
                    ),
                )
            )
    return tuple(result)


def analyze_value_lifetimes(
    module: Module,
    function: Function,
    *,
    topology_level: str | None = None,
) -> tuple[ValueLifetime, ...]:
    """Project checked structural SSA liveness into memory residency."""
    liveness = analyze_liveness(function)
    facts = module.resolve_target().get_facts(MemoryHierarchyFacts)
    local = CostContext(
        scope=FunctionScope(module, function),
        topology_level=topology_level,
        topologies=module.effective_topologies(),
    )
    projected = _project_allocation_values(
        liveness,
        _resident_value_ids(function, liveness),
        frozenset(id(parameter) for parameter in function.params),
        facts,
        local,
    )
    return tuple(item.lifetime for item in projected)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    whole: CostContext | None = None
    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    storage: _TrafficAccounts = field(default_factory=_TrafficAccounts)
    communication: _TrafficAccounts = field(default_factory=_TrafficAccounts)


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic and aggregate it over each enclosing loop."""

    def visit_LoopRegion(self, expr: LoopRegion, ctx: MemoryContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.init_args:
            self.visit(operand, ctx)
        self.visit(expr.body, inner)
        for operand in expr.yield_values:
            self.visit(operand, inner)

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: MemoryContext
    ) -> None:
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
            else MemoryMetadata()
        )
        attach(expr, moved)
        if not recorded:
            return
        logical_repeats = 1
        total_repeats = 1
        cursor = ctx.current
        while cursor.parent is not None:
            trips = max(1, cursor.trips())
            total_repeats *= trips
            if cursor.is_variant(expr):
                logical_repeats *= trips
            cursor = cursor.parent
        add_traffic(
            ctx.storage,
            ctx.communication,
            moved,
            logical_repeats,
            total_repeats,
        )


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Attach Call movement and Function-wide movement and placement."""
    module = context.module
    topology_level = context.topology_level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    topologies = module.effective_topologies()
    whole = CostContext(scope=FunctionScope(module, function))
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
        locals_by_unit=locals_by_unit,
    )
    MemoryVisitor().visit(function.body, memory_context)
    liveness = analyze_liveness(function)
    placement = CostContext(
        scope=FunctionScope(module, function),
        topology_level=topology_level,
        topologies=topologies,
    )
    allocation_values = _project_allocation_values(
        liveness,
        _resident_value_ids(function, liveness),
        frozenset(id(parameter) for parameter in function.params),
        facts,
        placement,
    )
    lifetimes = tuple(item.lifetime for item in allocation_values)
    solver_options = (
        context.options if isinstance(context.options, MemoryOptions) else MemoryOptions()
    )
    levels_list: list[MemoryLevelPeak] = []
    for name in sorted(
        {item.memory_level for item in lifetimes} | set(memory_context.storage.total)
    ):
        declared = facts.explicit(name)
        values = tuple(item for item in allocation_values if item.lifetime.memory_level == name)
        rows = [item.lifetime for item in values]
        capacity = declared.capacity_bytes if declared is not None else None
        if name in (str(StorageKind.GMEM), str(StorageKind.SMEM)) and values:
            solved = solve_allocation(
                name,
                values,
                liveness,
                context.root,
                options=solver_options,
            )
            peak = solved.peak_bytes
        elif name == str(StorageKind.RMEM):
            peak = max((item.bytes for item in rows), default=0)
        else:
            peak = 0
            end = max((item.last_used_at for item in rows), default=-1)
            for point in range(end + 1):
                peak = max(
                    peak,
                    sum(
                        item.bytes for item in rows if item.defined_at <= point <= item.last_used_at
                    ),
                )
        levels_list.append(
            MemoryLevelPeak(
                memory_level=name,
                peak_bytes=peak,
                persistent_bytes=sum(item.bytes for item in rows if item.persistent),
                capacity_bytes=capacity,
            )
        )
    levels = tuple(levels_list)
    errors = tuple(
        f"{item.memory_level} placement peak {item.peak_bytes} B exceeds "
        f"capacity {item.capacity_bytes} B"
        for item in levels
        if item.exceeds_capacity
    )
    attach(
        function,
        RegionMemoryMetadata(
            topologies=tuple(locals_by_unit),
            traffic=Traffic(
                storage=_account_shares(memory_context.storage, tuple(locals_by_unit)),
                communication=_account_shares(memory_context.communication, tuple(locals_by_unit)),
            ),
            lifetimes=lifetimes,
            peaks=levels,
            solver_status="feasible",
            errors=errors,
        ),
    )


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
    "analyze_value_lifetimes",
]
