"""Memory-family projection from shared IterationScope and Access records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import (
    Call,
    Expr,
    RangeMetadata,
    Var,
    VerifyError,
    describe_expr,
    get_metadata,
    value_labels,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.mesh import Mesh, separate, within_scope
from tilefoundry.ir.types.shard_layout import shard_layout_of
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.utils import (
    bytes_by_storage,
    is_unsharded_layout,
    participant_layout,
    static_dim_value,
)
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.utils.units import format_bytes
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    leaves_of,
    projected,
    reached_elements,
    reached_leaves,
    relations_of,
    static_bytes,
)
from tilefoundry.visitor_registry.call_frame import CallFrame
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .allocation import (
    AllocationConstraintVisitor,
    AllocationModel,
    AllocationValue,
    aligned,
    alignment_of,
    solve_allocation,
)
from .errors import AnalysisError
from .facts import TARGET_MEMORY_OWNER, MemoryHierarchyFacts
from .footprint import (
    ReachedAddresses,
    cached_level,
    footprint_of,
    merged,
    reached_by,
    reuse_windows,
    wave_of,
)
from .invocation import InvocationWalker, invocation_key
from .iteration_scope import IterationScope, Repeats, walk_scopes
from .liveness import analyze_liveness, result_copies
from .metadata import (
    Breakdown,
    MemoryLevelPeak,
    MemoryMetadata,
    RegionMemoryMetadata,
    Spread,
    Traffic,
    TrafficBytes,
)
from .precision import AnalysisPrecision
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
    for held, access in boundaries:
        leaves = leaves_of(held)
        if not leaves:
            return None
        if len(leaves) == 1:
            taken = {0: _bytes_for(leaves[0], reached_elements(access))}
        else:
            reached = reached_leaves(access, len(leaves))
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
    scope: IterationScope | None = None,
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
        local_relations = (
            scope.projected_relations(call, ctx)
            if scope is not None
            else projected(relations_of(call, ctx), call, ctx)
        )
        result = ctx.local_type_of(call)
        fields = result.fields if isinstance(result, TupleType) else (result,)
        outputs = local_relations[len(call.args) :]
        if len(fields) > len(outputs):
            raise AnalysisError(
                f"{describe_expr(call)}: states {len(outputs)} output "
                f"boundaries for a result of {len(fields)} fields"
            )
        amounts, charged = [], []
        for index, moved in enumerate(cost.traffic):
            if index == len(call.args):
                asked = tuple(
                    (field_, outputs[position])
                    for position, field_ in enumerate(fields)
                )
                memory_level = None
            else:
                asked = ((types[index], local_relations[index]),)
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
    scope: IterationScope | None = None,
    counted: str | None = None,
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
        levels, positional = _movement(expr, cost, ctx, types, scope)
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
    positions = scope.mesh_positions if scope is not None else lambda _unit: 1
    counted = counted or next(iter(locals_by_unit))
    return MemoryMetadata(
        topologies=tuple(locals_by_unit),
        traffic=Traffic(
            storage=_occurrence_shares(
                storage_whole, storage_per_unit, tuple(locals_by_unit), counted, positions
            ),
            communication=_occurrence_shares(
                crossing_whole, crossing_per_unit, tuple(locals_by_unit), None, positions
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
    counted: str | None,
    positions: Callable[[str | None], int],
) -> "Breakdown[TrafficBytes]":
    """One entry per level, each carrying the whole and every level's share.

    ``counted`` selects storage's unit; None counts each communication boundary
    at its own topology. Total scales that share by its executing positions.
    The shares run in *topologies* order, so a level a move never reached --
    a unit inside the boundary it crossed -- reads as no bytes rather than as
    a missing entry.
    """
    entries = []
    for name in sorted({*whole, *per_unit}):
        unit = counted if counted is not None else name
        if unit not in topologies:
            raise AnalysisError(f"memory: counted topology {unit!r} is not declared")
        moved = per_unit.get(name, {}).get(unit, TrafficBytes())
        factor = positions(unit)
        entries.append(
            (
                name,
                Spread(
                    logical=whole.get(name, TrafficBytes()),
                    total=TrafficBytes(moved.read * factor, moved.write * factor),
                    per_unit=tuple(
                        per_unit.get(name, {}).get(level, TrafficBytes()) for level in topologies
                    ),
                ),
            )
        )
    return Breakdown(tuple(entries))


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
    repeats: Repeats,
) -> None:
    """Add one occurrence to each independent Function counting domain."""
    for into, stated in (
        (storage, record.traffic.storage),
        (communication, record.traffic.communication),
    ):
        for name, spread in stated.kinds:
            _accumulate(into.logical, name, spread.logical, repeats.varying_loop_trips)
            _accumulate(into.total, name, spread.total, repeats.loop_trips)
            for unit, moved in zip(record.topologies, spread.per_unit, strict=False):
                _accumulate(into.per_unit.setdefault(name, {}), unit, moved, repeats.loop_trips)


def coalesced_peak_in_window(
    values: tuple[AllocationValue, ...],
    model: AllocationModel,
    entered_at: int,
    exited_at: int,
) -> int:
    """Sum one physical buffer per required-alias group at each event."""
    peak = 0
    for point in range(entered_at, exited_at + 1):
        live: dict[AllocationValue, int] = {}
        for item in values:
            if item.lifetime.persistent or (
                item.lifetime.defined_at <= point <= item.lifetime.last_used_at
            ):
                group = model.groups.find(item)
                live[group] = max(live.get(group, 0), item.lifetime.bytes)
        peak = max(peak, sum(live.values()))
    return peak


def held_in_region(value_mesh: Mesh, region_mesh: Mesh) -> bool:
    """Whether the region selects holders of *value_mesh* at every level it names.

    A value mesh may also name enclosing levels. Those do not reject a region
    that states only an inner level; each region level is compared independently
    so sliced offsets remain in that topology's own numbering.
    """
    value_levels = {
        getattr(level.topologies[0], "name", level.topologies[0]): level
        for level in separate(value_mesh)
    }
    for region_level in separate(region_mesh):
        name = getattr(region_level.topologies[0], "name", region_level.topologies[0])
        value_level = value_levels.get(name)
        if value_level is None or not within_scope(value_level, region_level):
            return False
    return True


def values_in_region(
    values: tuple[AllocationValue, ...],
    region: MeshRegion,
    entered_at: int,
    exited_at: int,
    facts: MemoryHierarchyFacts,
    topology_names: tuple[str, ...],
) -> tuple[AllocationValue, ...]:
    """Resident values whose lifetimes and ownership intersect one mesh region.

    Unsharded tensor leaves defined in the window contribute while live,
    including definitions in child regions. Alias groups are counted once.
    """
    positions = {name: index for index, name in enumerate(topology_names)}
    region_positions = tuple(
        positions[getattr(topology, "name", topology)] for topology in region.mesh.topologies
    )
    outermost_region = min(region_positions)
    result = []
    for item in values:
        lifetime = item.lifetime
        if (not lifetime.persistent and lifetime.last_used_at < entered_at) or lifetime.defined_at > exited_at:
            continue
        declared = facts.explicit(lifetime.memory_level)
        if declared is None or not declared.owner:
            raise AnalysisError(f"memory: level {lifetime.memory_level!r} has no declared owner")
        if declared.owner == TARGET_MEMORY_OWNER:
            continue
        owner_position = positions.get(declared.owner)
        if owner_position is None:
            raise AnalysisError(
                f"memory: level {lifetime.memory_level!r} owner {declared.owner!r} "
                f"is not a declared topology in {topology_names}"
            )
        held = leaves_of(item.value.type)[item.index]
        if isinstance(held, TensorType) and is_unsharded_layout(held.layout):
            if not entered_at < lifetime.defined_at < exited_at:
                continue
        else:
            if outermost_region > owner_position:
                continue
            layout = shard_layout_of(getattr(held, "layout", None))
            if layout is None or not held_in_region(participant_layout(layout).mesh, region.mesh):
                continue
        result.append(item)
    return tuple(result)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    whole: CostContext | None = None
    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    storage: _TrafficAccounts = field(default_factory=_TrafficAccounts)
    communication: _TrafficAccounts = field(default_factory=_TrafficAccounts)
    memory_level: str | None = None
    wave: tuple[int, int] | None = None
    reached_by_scope: dict[
        IterationScope, list[tuple[Call, tuple[ReachedAddresses, ...]]]
    ] = field(default_factory=dict)
    call_reached: list[tuple[IterationScope, Call, tuple[ReachedAddresses, ...]]] = field(
        default_factory=list
    )
    footprint_available: bool = False


def _footprint_inputs(
    root: IterationScope,
    *,
    memory_level: str,
    wave: tuple[int, int],
    whole: CostContext,
) -> tuple[
    dict[IterationScope, list[tuple[Call, tuple[ReachedAddresses, ...]]]],
    bool,
]:
    """Account for Calls with no recorded boundaries."""
    refused: dict[IterationScope, list[tuple[Call, tuple[ReachedAddresses, ...]]]] = {}
    available = True
    wave_units, declared_units = wave
    for scope in walk_scopes(root):
        for call in scope.refused.get("narrow", ()):
            reached = reached_by(
                scope,
                call,
                memory_level=memory_level,
                wave_units=wave_units,
                declared_units=declared_units,
                operands=(),
                ctx=whole,
                window=scope.depth - 1,
            )
            if reached is None:
                available = False
            else:
                refused.setdefault(scope, []).append((call, reached))

    return refused, available


class MemoryVisitor(InvocationWalker, ExprVisitor[None]):
    """Attach per-Call traffic and aggregate it over each enclosing loop."""

    def call_result(self, expr, ctx):
        self.attach(expr, MemoryMetadata(), ctx)

    def visit_MeshRegion(self, expr: MeshRegion, ctx: MemoryContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        for arg in expr.args:
            self.visit(arg, ctx)
        self.visit(expr.body, replace(ctx, current=child))

    def visit_LoopRegion(self, expr: LoopRegion, ctx: MemoryContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.args:
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
            call_traffic(expr, ctx.whole, ctx.locals_by_unit, ctx.current, ctx.topology_level)
            if recorded
            else MemoryMetadata()
        )
        if recorded and ctx.wave is not None and ctx.memory_level is not None:
            reached = reached_by(
                ctx.current,
                expr,
                memory_level=ctx.memory_level,
                wave_units=ctx.wave[0],
                declared_units=ctx.wave[1],
                operands=moved.operands,
                ctx=ctx.whole,
                window=ctx.current.depth - 1,
            )
            if reached is not None:
                ctx.reached_by_scope.setdefault(ctx.current, []).append((expr, reached))
                ctx.call_reached.append((ctx.current, expr, reached))
            else:
                ctx.footprint_available = False
        self.attach(expr, moved, ctx)
        if not recorded:
            return
        repeats = ctx.current.repeats_of(expr, ctx.topology_level)
        add_traffic(
            ctx.storage,
            ctx.communication,
            moved,
            repeats,
        )


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Attach Call movement and Function-wide movement and placement."""
    module = context.module
    topology_level = context.topology_level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    owner_topologies = tuple(
        level.name for level in context.target.get_facts(TopologyFacts).topologies
    )
    for level in facts.explicit_levels:
        if not level.owner:
            raise AnalysisError(f"memory: level {level.name!r} has no declared owner")
        if level.owner != TARGET_MEMORY_OWNER and level.owner not in owner_topologies:
            raise AnalysisError(
                f"memory: level {level.name!r} owner {level.owner!r} is not a "
                f"target topology in {owner_topologies}"
            )
    topologies = module.effective_topologies()
    whole = CostContext(frame=CallFrame(module, function))
    cache = cached_level(facts)
    wave = wave_of(module, context.target, topology_level) if cache is not None else None
    memory_level = cache[1] if cache is not None and wave is not None else None
    refused_reached: dict[IterationScope, list[tuple[Call, tuple[ReachedAddresses, ...]]]] = {}
    footprint_available = False
    if memory_level is not None and wave is not None:
        (
            refused_reached,
            footprint_available,
        ) = _footprint_inputs(
            context.root,
            memory_level=memory_level,
            wave=wave,
            whole=whole,
        )
    units = tuple(topology.name for topology in topologies) or (
        (topology_level,) if topology_level else ()
    )
    locals_by_unit = {
        unit: CostContext(
            frame=CallFrame(module, function),
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
        records=context.records,
        whole=whole,
        locals_by_unit=locals_by_unit,
        memory_level=memory_level,
        wave=wave,
        reached_by_scope=refused_reached,
        footprint_available=footprint_available,
    )
    MemoryVisitor().visit(function.body, memory_context)
    merged_reached = merged(
        item
        for calls in memory_context.reached_by_scope.values()
        for _call, reached in calls
        for item in reached
    )
    distinct: dict[tuple[int, int], Expr] = {}
    for item in merged_reached:
        if item.reached is not None:
            distinct.setdefault(
                (invocation_key(item.invocation), id(item.buffer), item.index), item.buffer
            )
    captured_names = {}
    for scope in walk_scopes(context.root):
        for param, _argument in scope.captures:
            captured_names.setdefault((scope.invocation.key, id(scope.capture_root(param))), param)
    label_sources = {(key[0], key[1]): value for key, value in distinct.items()}
    label_values = (
        value if getattr(value, "name", None) else captured_names.get(key, value)
        for key, value in label_sources.items()
    )
    source_labels = dict(zip(label_sources, value_labels(label_values), strict=True))
    footprint_labels = {
        key: (f"{key[0].label}/" if key[0] is not None else "")
        + source_labels[key[:2]]
        + (f"[{key[2]}]" if isinstance(value.type, TupleType) else "")
        for key, value in distinct.items()
    }
    liveness = analyze_liveness(function, context.root.invocation)
    placement = CostContext(
        frame=CallFrame(module, function),
        topology_level=topology_level,
        topologies=topologies,
    )
    quantities = {}
    persistent = set()
    resources = {}

    def multiplicity(invocation):
        count = 1
        while invocation.caller is not None:
            caller = invocation.caller
            if not isinstance(invocation.index, Expr):
                invocation = caller
                continue
            slot = caller.frame.module.children[invocation.slot_key]
            cursor = invocation
            number = None
            while cursor is not None:
                loop = cursor.owner
                if isinstance(loop, LoopRegion) and loop.induction_var is invocation.index:
                    start, stop, step = (
                        static_dim_value(value) for value in (loop.start, loop.extent, loop.step)
                    )
                    if start is not None and stop is not None and step is not None and step != 0:
                        trips = len(range(start, stop, step))
                        if step > 0:
                            first = max(0, -(start // step))
                            last = min(trips, -(-(len(slot.elements) - start) // step))
                        else:
                            stride = -step
                            first = max(0, (start - len(slot.elements)) // stride + 1)
                            last = min(trips, start // stride + 1)
                        number = max(0, last - first)
                    break
                cursor = cursor.parent
            if number is None:
                bounds = get_metadata(invocation.index, RangeMetadata)
                number = (
                    len(slot.elements) if bounds is None
                    else max(0, min(len(slot.elements), bounds.hi) - max(0, bounds.lo))
                )
            count *= number
            invocation = caller
        return count

    limits: dict[str, int] = {}
    for interval in liveness.intervals:
        invocation = interval.invocation
        key = (invocation_key(invocation), id(interval.value))
        multiplier = 1
        if invocation is not None and invocation.parent is not None and any(interval.value is weight for weight in invocation.weights):
            resource = invocation.resource_key(interval.value)
            multiplier = multiplicity(invocation)
            resources[key] = resource
            persistent.add(key)
        elif invocation is None or invocation.parent is None:
            if any(interval.value is param for param in function.params):
                persistent.add(key)
        held = placement.local_type_of(interval.value)
        slots = []
        for leaf in leaves_of(held):
            quantity = next(
                ((level, amount) for level, amount in bytes_by_storage(leaf).items()
                 if facts.explicit(level) is not None),
                None,
            )
            if quantity is None:
                slots.append(None)
                continue
            level, amount = quantity
            amount *= result_copies(interval.value) * multiplier
            slots.append((level, amount))
            limits[level] = limits.get(level, 0) + aligned(amount, alignment_of(interval.value))
        quantities[key] = tuple(slots)
    model = AllocationModel(
        current=context.root,
        liveness=liveness,
        local=placement,
        quantities=quantities,
        persistent=frozenset(persistent),
        limits=limits,
        records=context.records,
    )
    allocation = AllocationConstraintVisitor(root_function=function, liveness=liveness)

    allocation.visit_function_body(function, model)
    definitions = {
        (invocation_key(interval.invocation), id(interval.value)): index
        for index, interval in enumerate(liveness.intervals)
    }
    model.values.sort(
        key=lambda item: (
            definitions[invocation_key(item.invocation), id(item.value)],
            item.index,
        )
    )
    allocation_values = tuple(model.values)
    material_values = tuple(item for item in allocation_values if item in model.material)
    producers = {
        (invocation_key(item.invocation), id(item.value)): item.value
        for item in (*material_values, *allocation_values)
    }
    labels = dict(zip(producers, value_labels(producers.values()), strict=True))
    for item in allocation_values:
        key = (invocation_key(item.invocation), id(item.value))
        label = resources.get(
            key, (f"{item.invocation.label}/" if item.invocation and item.invocation.parent else "") + labels[key]
        )
        item.lifetime = replace(
            item.lifetime,
            binding=label + (f"[{item.index}]" if isinstance(item.value.type, TupleType) else ""),
        )
    lifetimes = tuple(item.lifetime for item in material_values)
    reuse = (
        reuse_windows(
            context.root,
            memory_level=memory_level,
            wave_units=wave[0],
            declared_units=wave[1],
            ctx=locals_by_unit.get(topology_level, whole),
            labels=footprint_labels,
        )
        if memory_level is not None and wave is not None
        else ()
    )
    if memory_level is not None and wave is not None:
        for scope, call, reached in memory_context.call_reached:
            repeats = scope.repeats_of(call, topology_level)
            moved = context.records.get(scope.invocation, call, MemoryMetadata)
            if moved is None:
                raise AnalysisError("memory: Call footprint has no movement record")
            context.records.attach(
                scope.invocation,
                call,
                replace(
                    moved,
                    footprint=footprint_of(
                        merged(reached),
                        memory_level=memory_level,
                        labels=footprint_labels,
                        windows=-(-repeats.mesh_positions // wave[0]),
                    ),
                ),
            )
    solver_options = (
        context.options if isinstance(context.options, MemoryOptions) else MemoryOptions()
    )
    region_rmem_peaks: list[int] = []
    claimed_rmem: set[AllocationValue] = set()
    for window in liveness.regions:
        region_values = values_in_region(
            allocation_values,
            window.region,
            window.entered_at,
            window.exited_at,
            facts,
            owner_topologies,
        )
        region_levels = []
        for name in sorted({item.lifetime.memory_level for item in region_values}):
            resident = tuple(item for item in region_values if item.lifetime.memory_level == name)
            declared = facts.explicit(name)
            peak = coalesced_peak_in_window(
                resident, model, window.entered_at, window.exited_at,
            )
            persistent = {
                model.groups.find(item): item.lifetime.bytes
                for item in resident if item.lifetime.persistent
            }
            region_levels.append(
                MemoryLevelPeak(
                    memory_level=name,
                    peak_bytes=peak,
                    persistent_bytes=sum(persistent.values()),
                    capacity_bytes=(declared.capacity_bytes if declared is not None else None),
                ),
            )
            if name == str(StorageKind.RMEM):
                region_rmem_peaks.append(peak)
                claimed_rmem.update(resident)
        if region_levels:
            context.records.attach(
                window.invocation,
                window.region,
                RegionMemoryMetadata(
                    solver_status="feasible",
                    topologies=tuple(locals_by_unit),
                    peaks=tuple(region_levels),
                ),
            )

    offsets: dict[AllocationValue, int] = {}
    levels_list: list[MemoryLevelPeak] = []
    for name in sorted(
        {item.lifetime.memory_level for item in allocation_values} | set(memory_context.storage.total)
    ):
        declared = facts.explicit(name)
        values = tuple(item for item in allocation_values if item.lifetime.memory_level == name)
        capacity = declared.capacity_bytes if declared is not None else None
        if name in (str(StorageKind.GMEM), str(StorageKind.SMEM)) and values:
            solved = solve_allocation(name, model, solver_options)
            offsets.update(solved.offsets)
            peak = solved.peak_bytes
            for item in values:
                if item not in model.material or item.lifetime.persistent:
                    continue
                if not isinstance(item.value, (Call, Var)) or isinstance(item.value.type, TupleType):
                    continue
                copies = result_copies(item.value)
                buffer_bytes = item.lifetime.bytes // copies
                moved = context.records.get(item.invocation, item.value, MemoryMetadata)
                if isinstance(item.value, Var) and moved is None:
                    moved = MemoryMetadata()
                if moved is None:
                    raise AnalysisError("memory: placed Call has no movement record")
                context.records.attach(
                    item.invocation,
                    item.value,
                    replace(
                        moved,
                        buffer_bytes=buffer_bytes,
                        offsets=tuple(
                            solved.offsets[item] + copy * buffer_bytes for copy in range(copies)
                        ),
                    ),
                )
        elif name == str(StorageKind.RMEM):
            unclaimed = tuple(item for item in values if item not in claimed_rmem)
            peak = max(
                max(region_rmem_peaks, default=0),
                coalesced_peak_in_window(unclaimed, model, 0, liveness.timeline_end),
            )
        else:
            peak = coalesced_peak_in_window(values, model, 0, liveness.timeline_end)
        persistent = {
            model.groups.find(item): item.lifetime.bytes for item in values if item.lifetime.persistent
        }
        levels_list.append(
            MemoryLevelPeak(
                memory_level=name,
                peak_bytes=peak,
                persistent_bytes=sum(persistent.values()),
                capacity_bytes=capacity,
            ),
        )
    levels = tuple(levels_list)
    tuple_results = {
        (item.invocation, id(item.value)): item.value
        for item in material_values
        if isinstance(item.value, Call) and isinstance(item.value.type, TupleType)
    }
    for (invocation, _identity), producer in tuple_results.items():
        moved = context.records.get(invocation, producer, MemoryMetadata)
        if moved is None:
            raise AnalysisError("memory: placed Call has no movement record")
        context.records.attach(
            invocation,
            producer,
            replace(
                moved,
                buffer_bytes=None,
                offsets=tuple(
                    offsets[item]
                    for item in material_values
                    if item.invocation is invocation and item.value is producer and item in offsets
                ),
            ),
        )
    shared = {}
    for item in allocation_values:
        invocation = item.invocation
        if (
            invocation is None
            or invocation.parent is None
            or any(item.value is param for param in invocation.function.params)
            or (invocation.result_tied and item.value is invocation.function.body)
        ):
            continue
        if item.lifetime.memory_level == "gmem" or item.lifetime.persistent:
            continue
        moved = context.records.get(invocation, item.value, MemoryMetadata)
        alias = None if moved is None else moved.alias
        alias_key = (
            None
            if alias is None
            else (
                id(alias.buffer),
                alias.index,
                str(alias.relation.relation),
                tuple((name, id(value)) for name, value in alias.relation.values.items()),
            )
        )
        state = None if moved is None else (moved.offsets, moved.buffer_bytes, alias_key)
        key = (id(item.value), item.index)
        if key in shared and shared[key][1] != state:
            previous, value = shared[key]
            raise AnalysisError(
                f"callee {invocation.function.name!r} node {item.lifetime.binding!r} has incompatible placement at {previous.label!r} and {invocation.label!r}: {value!r} versus {state!r}"
            )
        shared[key] = invocation, state
    errors = tuple(
        f"{item.memory_level} placement peak {format_bytes(item.peak_bytes)} exceeds "
        f"capacity {format_bytes(item.capacity_bytes)}"
        for item in levels
        if item.exceeds_capacity
    )
    advisories: tuple[str, ...] = ()
    reliable = (AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND)
    overfull_snapshot_holds: set[tuple[int, bool]] = set()
    if cache is not None and wave is not None:
        cache_level, _backing_level, cache_capacity_bytes = cache
        overfull_windows: dict[str, tuple[int, AnalysisPrecision]] = {}
        for row in reuse:
            if not row.fits:
                window = row.time or row.space
                overfull_windows.setdefault(window, (row.holds_bytes, row.precision))
                if not row.time:
                    overfull_snapshot_holds.add((row.holds_bytes, row.precision in reliable))
        for window, (holds_bytes, precision) in overfull_windows.items():
            message = (
                f"{cache_level} reuse window {window} holds {format_bytes(holds_bytes)} at a "
                f"{wave[0]}-unit wave, exceeding capacity "
                f"{format_bytes(cache_capacity_bytes)}"
            )
            if precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
    footprint = None
    cache_level = ""
    cache_capacity_bytes = None
    wave_units = 0
    declared_units = 0
    if (
        cache is not None
        and wave is not None
        and memory_level is not None
        and memory_context.footprint_available
    ):
        footprint = footprint_of(
            merged_reached,
            memory_level=memory_level,
            labels=footprint_labels,
        )
        totals: dict[tuple[str, str], int] = {}
        for scope, calls in memory_context.reached_by_scope.items():
            repeats = scope.repeats_of(calls[0][0], topology_level)
            counted = footprint_of(
                merged(item for _call, reached in calls for item in reached),
                memory_level=memory_level,
                labels=footprint_labels,
                windows=-(-repeats.mesh_positions // wave[0]) * repeats.loop_trips,
            )
            for buffer, breakdown in counted.buffers:
                for level, spread in breakdown.kinds:
                    key = (buffer, level)
                    totals[key] = totals.get(key, 0) + spread.total
        footprint = replace(
            footprint,
            buffers=tuple(
                (
                    buffer,
                    Breakdown(
                        tuple(
                            (level, replace(spread, total=totals[(buffer, level)]))
                            for level, spread in breakdown.kinds
                        )
                    ),
                )
                for buffer, breakdown in footprint.buffers
            ),
        )
        cache_level, _backing_level, cache_capacity_bytes = cache
        wave_units, declared_units = wave
        used = sum(
            spread.logical
            for _buffer, breakdown in footprint.buffers
            for _level, spread in breakdown.kinds
        )
        if (
            used >= cache_capacity_bytes
            and (used, footprint.precision in reliable) not in overfull_snapshot_holds
        ):
            message = (
                f"{cache_level} working set {format_bytes(used)} at the first iteration "
                f"of a {wave_units}-unit wave exceeds capacity "
                f"{format_bytes(cache_capacity_bytes)}"
            )
            if footprint.precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
    attach(
        function,
        RegionMemoryMetadata(
            topologies=tuple(locals_by_unit),
            traffic=Traffic(
                storage=_account_shares(memory_context.storage, tuple(locals_by_unit)),
                communication=_account_shares(memory_context.communication, tuple(locals_by_unit)),
            ),
            footprint=footprint,
            reuse_windows=reuse,
            lifetimes=lifetimes,
            peaks=levels,
            solver_status="feasible",
            errors=errors,
            advisories=advisories,
        ),
    )


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
]
