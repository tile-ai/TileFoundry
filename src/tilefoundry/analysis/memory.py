"""Memory-family projection from shared IterationScope and Access records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.core import (
    Call,
    Expr,
    VerifyError,
    binding_name,
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
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .allocation import (
    AllocationConstraintVisitor,
    AllocationModel,
    AllocationValue,
    Buffer,
    leaves,
    published_aliases,
    solve_allocation,
    tie_analysis,
)
from .compute_cost import accumulate_logical, grouped_breakdown
from .errors import AnalysisError
from .facts import TARGET_MEMORY_OWNER, MemoryHierarchyFacts
from .footprint import (
    ReachedAddresses,
    cached_level,
    footprint_of,
    merged,
    reached_by,
    wave_of,
)
from .iteration_scope import IterationScope, Repeats, walk_scopes
from .liveness import analyze_liveness, result_copies
from .metadata import (
    Breakdown,
    MemoryLevelPeak,
    MemoryMetadata,
    RegionMemoryMetadata,
    Spread,
    TimingValuation,
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
                asked = tuple((field_, outputs[position]) for position, field_ in enumerate(fields))
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
    groups: dict = field(default_factory=dict)
    total: dict[str, TrafficBytes] = field(default_factory=dict)
    per_unit: dict[str, dict[str, TrafficBytes]] = field(default_factory=dict)


def _account_shares(
    accounts: _TrafficAccounts, topologies: tuple[str, ...]
) -> Breakdown[TrafficBytes]:
    """Build a breakdown without encoding an account as a topology key."""
    names = sorted({*accounts.logical, *accounts.total, *accounts.per_unit})
    return grouped_breakdown(
        Breakdown(
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
        ),
        accounts.groups,
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
    expr: Call,
    ctx: MemoryContext,
    binding=None,
) -> None:
    """Add one occurrence to each independent Function counting domain."""
    for into, stated in (
        (storage, record.traffic.storage),
        (communication, record.traffic.communication),
    ):
        for name, spread in stated.kinds:
            accumulate_logical(into.groups, name, spread, expr, ctx, binding)
            into.logical.clear()
            for (kind, _ids), (_formals, amount) in into.groups.items():
                _accumulate(into.logical, kind, amount, 1)
            _accumulate(into.total, name, spread.total, repeats.loop_trips)
            for unit, moved in zip(record.topologies, spread.per_unit, strict=False):
                _accumulate(into.per_unit.setdefault(name, {}), unit, moved, repeats.loop_trips)


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
    including definitions in child regions.
    """
    positions = {name: index for index, name in enumerate(topology_names)}
    region_positions = tuple(
        positions[getattr(topology, "name", topology)] for topology in region.mesh.topologies
    )
    outermost_region = min(region_positions)
    result = []
    for item in values:
        lifetime = item.lifetime
        if (
            not lifetime.persistent and lifetime.last_used_at < entered_at
        ) or lifetime.defined_at > exited_at:
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
    reached_by_scope: dict[IterationScope, list[tuple[Call, tuple[ReachedAddresses, ...]]]] = field(
        default_factory=dict
    )
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


class MemoryVisitor(ExprVisitor[None]):
    """Attach per-Call traffic and aggregate it over each enclosing loop."""

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
        callee = CallFrameVisitor.callee_frame(expr, ctx.frame)
        binding = None
        if callee is None:
            moved = (
                call_traffic(expr, ctx.whole, ctx.locals_by_unit, ctx.current, ctx.topology_level)
                if recorded
                else MemoryMetadata()
            )
            dependencies = tuple(
                formal
                for formal in ctx.root.owner.params
                if id(formal) in ctx.current.variance.get(id(expr), frozenset())
            )

            def grouped(held):
                return replace(
                    held,
                    kinds=tuple(
                        (name, replace(spread, logical_groups=((dependencies, spread.logical),)))
                        for name, spread in held.kinds
                    ),
                )

            moved = replace(
                moved,
                traffic=Traffic(
                    grouped(moved.traffic.storage), grouped(moved.traffic.communication)
                ),
            )
        else:
            summary = get_metadata(callee.function, RegionMemoryMetadata)
            if summary is None:
                raise AnalysisError(f"memory: missing summary for {callee.function.name!r}")
            moved = MemoryMetadata(
                topologies=summary.topologies, traffic=summary.traffic, footprint=summary.footprint
            )
            binding = ctx.call_context(expr)
        if callee is not None and moved.footprint is not None:
            from .footprint import resolve_footprint  # noqa: PLC0415

            footprint = resolve_footprint(moved.footprint, binding)
            if footprint is not None:
                ctx.reached_by_scope.setdefault(ctx.current, []).append((expr, footprint.reached))
        if callee is None and recorded and ctx.wave is not None and ctx.memory_level is not None:
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
        attach(expr, moved)
        if not recorded:
            return
        repeats = ctx.current.repeats_of(expr, ctx.topology_level)
        add_traffic(
            ctx.storage,
            ctx.communication,
            moved,
            repeats,
            expr,
            ctx,
            binding,
        )


def _analyze_memory(function: Function, context: AnalyzeContext) -> None:
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
    topologies = context.topologies
    whole = CostContext(frame=context.frame, current_mesh=context.root.execution_mesh())
    cache = cached_level(facts)
    wave = (
        wave_of(module, context.target, topology_level, topologies=topologies)
        if cache is not None
        else None
    )
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
            frame=context.frame,
            topology_level=unit,
            topologies=topologies,
        )
        for unit in units
    }
    memory_context = context.family(
        MemoryContext,
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
            distinct.setdefault((id(item.buffer), item.index), item.buffer)
    captured_names = {}
    for scope in walk_scopes(context.root):
        for param, _argument in scope.captures:
            captured_names.setdefault(id(scope.capture_root(param)), param)
    label_sources = {key[0]: value for key, value in distinct.items()}
    label_values = (
        value if getattr(value, "name", None) else captured_names.get(key, value)
        for key, value in label_sources.items()
    )
    source_labels = dict(zip(label_sources, value_labels(label_values), strict=True))
    footprint_labels = {
        key: source_labels[key[0]] + (f"[{key[1]}]" if isinstance(value.type, TupleType) else "")
        for key, value in distinct.items()
    }
    liveness = context.liveness = analyze_liveness(function, context)
    placement = CostContext(
        frame=context.frame,
        topology_level=topology_level,
        topologies=topologies,
    )
    quantities = {}
    persistent = set()

    for interval in liveness.intervals:
        key = id(interval.value)
        if any(interval.value is param for param in function.params):
            persistent.add(key)
        held = placement.local_type_of(interval.value)
        slots = []
        for leaf in leaves_of(held):
            quantity = next(
                (
                    (level, amount)
                    for level, amount in bytes_by_storage(leaf).items()
                    if facts.explicit(level) is not None
                ),
                None,
            )
            if quantity is None:
                slots.append(None)
                continue
            level, amount = quantity
            amount *= result_copies(interval.value)
            slots.append((level, amount))
        quantities[key] = tuple(slots)
    model = AllocationModel(
        current=context.root,
        liveness=liveness,
        local=placement,
        quantities=quantities,
        persistent=frozenset(persistent),
        context=context,
    )
    allocation = AllocationConstraintVisitor(root_function=function, liveness=liveness)

    returned = allocation.visit_function_body(function, model)
    model.returned = leaves(returned)
    model.parameters = {
        item for item in model.values if any(item.value is formal for formal in function.params)
    }
    model.outputs = set(model.returned)
    context.allocation = model
    definitions = {id(interval.value): index for index, interval in enumerate(liveness.intervals)}
    model.values.sort(
        key=lambda item: (
            definitions[id(item.value)],
            item.index,
        )
    )
    allocation_values = tuple(model.values)
    material_values = tuple(item for item in allocation_values if item in model.material)
    producers = {id(item.value): item.value for item in (*material_values, *allocation_values)}
    labels = dict(zip(producers, value_labels(producers.values()), strict=True))
    for item in allocation_values:
        key = id(item.value)
        label = labels[key]
        item.lifetime = replace(
            item.lifetime,
            binding=label + (f"[{item.index}]" if isinstance(item.value.type, TupleType) else ""),
        )
    lifetimes = tuple(item.lifetime for item in material_values)
    if memory_level is not None and wave is not None:
        seen_footprints = set()
        for scope, call, reached in memory_context.call_reached:
            if id(call) in seen_footprints:
                continue
            seen_footprints.add(id(call))
            repeats = scope.repeats_of(call, topology_level)
            moved = get_metadata(call, MemoryMetadata)
            if moved is None:
                raise AnalysisError("memory: Call footprint has no movement record")
            attach(
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
    model.definitions = dict(allocation.definitions)
    model.uses = {
        value: tuple(allocation.uses.get((id(value.value), value.index), ()))
        for value in model.values
    }
    tie_analysis(model)
    for node in {id(value.value): value.value for value in model.values}.values():
        moved = get_metadata(node, MemoryMetadata)
        if moved is not None:
            attach(node, replace(moved, alias=None, buffer_bytes=None, offsets=()))
    for value, alias in published_aliases(model).items():
        moved = get_metadata(value.value, MemoryMetadata) or MemoryMetadata()
        attach(value.value, replace(moved, alias=alias))
    footprint = None
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
        groups = []
        for scope, calls in memory_context.reached_by_scope.items():
            repeats = scope.repeats_of(calls[0][0], topology_level)
            counted = footprint_of(
                merged(item for _call, reached in calls for item in reached),
                memory_level=memory_level,
                labels=footprint_labels,
                windows=-(-repeats.mesh_positions // wave[0]) * repeats.loop_trips,
            )
            groups.extend(counted.groups)
            for buffer, breakdown in counted.buffers:
                for level, spread in breakdown.kinds:
                    key = (buffer, level)
                    totals[key] = totals.get(key, 0) + spread.total
        footprint = replace(
            footprint,
            groups=tuple(groups),
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
    attach(
        function,
        RegionMemoryMetadata(
            topologies=tuple(locals_by_unit),
            traffic=Traffic(
                storage=_account_shares(memory_context.storage, tuple(locals_by_unit)),
                communication=_account_shares(memory_context.communication, tuple(locals_by_unit)),
            ),
            footprint=footprint,
            lifetimes=lifetimes if context.root.is_root_definition else (),
            peaks=(),
            solver_status="feasible",
        ),
    )


def _memory_findings(peaks, footprint, reuse, cache, wave):
    """Build placement and cache findings once from absolute root conclusions."""
    errors = tuple(
        f"{peak.memory_level} placement peak {format_bytes(peak.peak_bytes)} exceeds "
        f"capacity {format_bytes(peak.capacity_bytes)}"
        for peak in peaks
        if peak.exceeds_capacity
    )
    advisories = ()
    if cache is not None and wave is not None and footprint is not None:
        reliable = (AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND)
        messages = {}
        for row in reuse:
            if not row.fits:
                messages.setdefault(row.time or row.space, (row.holds_bytes, row.precision))
        for window, (held, precision) in messages.items():
            message = (
                f"{cache[0]} reuse window {window} holds {format_bytes(held)} at a "
                f"{wave[0]}-unit wave, exceeding capacity {format_bytes(cache[2])}"
            )
            if precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
        used = sum(
            spread.logical for _name, held in footprint.buffers for _level, spread in held.kinds
        )
        if used >= cache[2] and not any(
            not row.fits
            and not row.time
            and row.holds_bytes == used
            and (row.precision in reliable) == (footprint.precision in reliable)
            for row in reuse
        ):
            message = (
                f"{cache[0]} working set {format_bytes(used)} at the first iteration of a "
                f"{wave[0]}-unit wave exceeds capacity {format_bytes(cache[2])}"
            )
            if footprint.precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
    return errors, advisories


def _global_allocation(context):
    """Join boundary instances; retain one address per non-boundary definition Buffer."""
    from .footprint import instance_image  # noqa: PLC0415
    from .visitor import VisitContext  # noqa: PLC0415

    visits = []
    bindings = {}
    shared = {}
    global_buffers = []
    weights = {}
    material_copies = []
    for local, visit in context.walk_visits():
        stack = visit.stack if isinstance(visit, VisitContext) else ()
        path = tuple(id(frame.call) for frame in stack)
        visits.append((local, visit, path))
        model = local.allocation
        timing = visit.request_valuation(TimingValuation)
        prefix = "/".join(binding_name(frame.call) or "call" for frame in stack)
        for buffer in model.buffers:
            boundary = any(
                value in model.parameters or value in model.outputs for value in buffer.members
            )
            key = (path, buffer) if boundary else (buffer,)
            placed = shared.get(key)
            if placed is None:
                placed = shared[key] = Buffer([])
                global_buffers.append(placed)
            bindings[path, buffer] = placed
            implicit = next(
                (
                    value
                    for value in buffer.members
                    if stack and any(value.value is formal for formal in stack[-1].implicit)
                ),
                None,
            )
            if implicit is not None:
                keys, image = instance_image(stack)
                reached = image.range()
                if reached.dim(isl.dim_type.PARAM):
                    reached = reached.project_out(
                        isl.dim_type.PARAM, 0, reached.dim(isl.dim_type.PARAM)
                    )
                key = keys, id(implicit.value), implicit.index
                if key not in weights:
                    weights[key] = (placed, implicit, reached)
                else:
                    original, value, previous = weights[key]
                    weights[key] = original, value, previous.union(reached)
                    bindings[path, buffer] = original
                    global_buffers.remove(placed)
                continue
            for item in buffer.members:
                life = item.lifetime
                start, end = timing.event(life.defined_at), timing.event(life.last_used_at)
                if result_copies(item.value) > 1:
                    for position, frame in enumerate(stack):
                        owner = frame.scope
                        while owner.parent is not None:
                            owner = owner.parent
                        caller = context.definitions[id(owner.owner)]
                        outer = VisitContext(stack[:position]) if position else None
                        for loop in frame.scope.enclosing_loops():
                            entered, exited = caller.liveness.loop_windows[id(loop)]
                            if outer is not None:
                                entered, exited = outer.event(entered), outer.event(exited)
                            start, end = min(start, entered), max(end, exited)
                translated = replace(
                    life,
                    binding=f"{prefix}/{life.binding}" if prefix else life.binding,
                    defined_at=start,
                    last_used_at=end,
                    persistent=life.persistent and not stack,
                )
                value = AllocationValue(item.value, item.index, translated)
                placed.members.append(value)
                if item in model.material and not (
                    stack and any(item.value is formal for formal, _argument in stack[-1].bindings)
                ):
                    material_copies.append(value)

    for (keys, _identity, _index), (buffer, item, reached) in weights.items():
        life = replace(
            item.lifetime,
            binding=".".join((*keys, item.lifetime.binding)),
            bytes=item.lifetime.bytes * reached.count_val().get_num_si(),
            defined_at=0,
            last_used_at=context.liveness.timeline_end,
            persistent=True,
        )
        value = AllocationValue(item.value, item.index, life)
        buffer.members.append(value)
        material_copies.append(value)

    def merge(left, right):
        if left is right:
            return
        right.members.extend(left.members)
        for key, value in tuple(bindings.items()):
            if value is left:
                bindings[key] = right
        global_buffers.remove(left)

    for local, visit, path in visits:
        if not path:
            continue
        stack = visit.stack
        parent = next(owner for owner, _valuation, key in visits if key == path[:-1])
        caller_model = parent.allocation
        model = local.allocation
        call_id = path[-1]
        for formal, actual in caller_model.call_bindings[call_id]:
            merge(
                bindings[path, model.partition[formal]],
                bindings[path[:-1], caller_model.partition[actual]],
            )
        for returned, result in zip(
            model.returned, caller_model.call_results[call_id], strict=True
        ):
            merge(
                bindings[path, model.partition[returned]],
                bindings[path[:-1], caller_model.partition[result]],
            )
    global_buffers = [buffer for buffer in global_buffers if buffer.members]
    for buffer in global_buffers:
        buffer.seal()
    root = context.allocation
    global_model = AllocationModel(
        root.current,
        root.liveness,
        root.local,
        {},
        frozenset(),
        buffers=global_buffers,
        values=[item for buffer in global_buffers for item in buffer.members],
    )
    placements = {}
    for local in context.definitions.values():
        model = local.allocation
        path = next(path for owner, _visit, path in visits if owner is local)
        for value in model.material:
            if value in model.parameters:
                continue
            buffer = model.partition[value]
            boundary = not local.root.is_root_definition and any(
                member in model.parameters or member in model.outputs for member in buffer.members
            )
            placements[value] = None if boundary else bindings[path, buffer]
    return global_model, placements, tuple(material_copies), visits


def _publish_placements(context, placements, offsets):
    """Publish global addresses only for local non-boundary material storage."""
    grouped = {}
    for value, buffer in placements.items():
        if value.lifetime.memory_level not in (str(StorageKind.GMEM), str(StorageKind.SMEM)):
            continue
        grouped.setdefault(id(value.value), (value.value, []))[1].append((value, buffer))
    for expr, entries in grouped.values():
        moved = get_metadata(expr, MemoryMetadata) or MemoryMetadata()
        entries.sort(key=lambda pair: pair[0].index)
        if isinstance(expr.type, TupleType):
            addresses = tuple(offsets[buffer] for _value, buffer in entries if buffer in offsets)
            attach(expr, replace(moved, buffer_bytes=None, offsets=addresses))
            continue
        value, buffer = entries[0]
        copies = result_copies(expr)
        size = value.lifetime.bytes // copies
        addresses = (
            tuple(offsets[buffer] + copy * size for copy in range(copies))
            if buffer in offsets and not value.lifetime.persistent
            else ()
        )
        attach(expr, replace(moved, buffer_bytes=size, offsets=addresses))


def _buffer_peaks(buffers, offsets, facts, *, resident=None, window=None):
    """Scan each event once, counting each Buffer's live bytes or address union."""
    levels = {}
    for buffer in buffers:
        for value in buffer.members:
            if resident is None or value in resident:
                levels.setdefault(value.lifetime.memory_level, []).append((buffer, value))
    result = []
    for name, items in sorted(levels.items()):
        events = {
            point
            for _buffer, value in items
            for point in (value.lifetime.defined_at, value.lifetime.last_used_at)
        }
        if window is not None:
            events = {
                window[0],
                window[1],
                *(point for point in events if window[0] <= point <= window[1]),
            }

        def occupied(point, persistent=False):
            live = {}
            for buffer, value in items:
                life = value.lifetime
                if life.persistent or (
                    not persistent and life.defined_at <= point <= life.last_used_at
                ):
                    live[buffer] = max(live.get(buffer, 0), life.bytes)
            if name not in (str(StorageKind.GMEM), str(StorageKind.SMEM)):
                return sum(live.values())
            spans = sorted(
                (offsets[buffer], offsets[buffer] + size) for buffer, size in live.items()
            )
            amount = 0
            end = 0
            for start, stop in spans:
                amount += max(0, stop - max(start, end))
                end = max(end, stop)
            return amount

        declared = facts.explicit(name)
        result.append(
            MemoryLevelPeak(
                name,
                max((occupied(point) for point in events), default=0),
                occupied(0, True),
                declared.capacity_bytes if declared else None,
            )
        )
    return tuple(result)


def _aggregate_root(function: Function, context: AnalyzeContext) -> None:
    """Publish only absolute root facts from composed call-stack visits."""
    from .footprint import root_footprint  # noqa: PLC0415

    record = get_metadata(function, RegionMemoryMetadata)
    facts = context.target.get_facts(MemoryHierarchyFacts)
    cache = cached_level(facts)
    wave = wave_of(
        context.module, context.target, context.topology_level, topologies=context.topologies
    )
    footprint, reuse = record.footprint, record.reuse_windows
    if cache is not None and wave is not None:
        footprint, reuse, _resources = root_footprint(context, memory_level=cache[1], wave=wave)
    global_model, placements, copies, visits = _global_allocation(context)
    options = context.options if isinstance(context.options, MemoryOptions) else MemoryOptions()
    global_model.addresses()
    offsets = {}
    placement_peaks = {}
    for level in sorted(global_model.models):
        solved = solve_allocation(level, global_model, options)
        offsets.update(solved.offsets)
        placement_peaks[level] = solved.peak_bytes
    _publish_placements(context, placements, offsets)
    lifetimes = tuple(item.lifetime for item in copies)
    peaks = tuple(
        replace(peak, peak_bytes=placement_peaks.get(peak.memory_level, peak.peak_bytes))
        for peak in _buffer_peaks(global_model.buffers, offsets, facts)
    )
    occupied = {peak.memory_level for peak in peaks}
    peaks += tuple(
        MemoryLevelPeak(
            name, 0, 0, facts.explicit(name).capacity_bytes if facts.explicit(name) else None
        )
        for name, _spread in record.traffic.storage.kinds
        if name not in occupied
    )
    peaks = tuple(sorted(peaks, key=lambda peak: peak.memory_level))
    region_peaks = {}
    topology_names = tuple(
        level.name for level in context.target.get_facts(TopologyFacts).topologies
    )
    claimed_rmem = set()
    rmem_peak = 0
    for local, visit, path in visits:
        timing = visit.request_valuation(TimingValuation)
        for window in local.liveness.regions:
            start, end = timing.event(window.entered_at), timing.event(window.exited_at)
            resident = values_in_region(
                tuple(global_model.values), window.region, start, end, facts, topology_names
            )
            levels = _buffer_peaks(
                global_model.buffers, offsets, facts, resident=set(resident), window=(start, end)
            )
            _region, held = region_peaks.setdefault(id(window.region), (window.region, {}))
            for level in levels:
                previous = held.get(level.memory_level)
                held[level.memory_level] = (
                    level
                    if previous is None
                    else replace(
                        level,
                        peak_bytes=max(previous.peak_bytes, level.peak_bytes),
                        persistent_bytes=max(previous.persistent_bytes, level.persistent_bytes),
                    )
                )
                if level.memory_level == str(StorageKind.RMEM):
                    rmem_peak = max(rmem_peak, level.peak_bytes)
                    claimed_rmem.update(
                        item
                        for item in resident
                        if item.lifetime.memory_level == str(StorageKind.RMEM)
                    )
    for region, held in region_peaks.values():
        attach(
            region,
            RegionMemoryMetadata(
                solver_status="feasible", topologies=record.topologies, peaks=tuple(held.values())
            ),
        )
    if claimed_rmem:
        unclaimed = set(global_model.values) - claimed_rmem
        others = _buffer_peaks(global_model.buffers, offsets, facts, resident=unclaimed)
        other_peak = next(
            (item.peak_bytes for item in others if item.memory_level == str(StorageKind.RMEM)), 0
        )
        peaks = tuple(
            replace(item, peak_bytes=max(rmem_peak, other_peak))
            if item.memory_level == str(StorageKind.RMEM)
            else item
            for item in peaks
        )
    errors, advisories = _memory_findings(peaks, footprint, reuse, cache, wave)
    attach(
        function,
        replace(
            record,
            footprint=footprint,
            reuse_windows=reuse,
            lifetimes=tuple(lifetimes),
            peaks=tuple(peaks),
            errors=errors,
            advisories=advisories,
        ),
    )


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Measure local definitions before aggregating the root call stack."""
    context.run_definitions(SELECTOR, _analyze_memory)
    _aggregate_root(function, context)


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
]
