"""Memory-family projection from shared IterationScope and Access records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import prod

from tilefoundry.ir.core import (
    Call,
    Expr,
    SSAValue,
    VerifyError,
    describe_expr,
    get_metadata,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.mesh import Mesh, separate, within_scope
from tilefoundry.ir.types.shard_layout import shard_layout_of
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.utils import (
    bytes_by_storage,
    is_unsharded_layout,
    leaves_of,
    participant_layout,
)
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.utils.units import format_bytes
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    projected,
    reached_elements,
    reached_leaves,
    relations_of,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .access import AliasKind
from .allocation import (
    AllocationConstraintVisitor,
    AllocationSummaryMetadata,
    solve_allocation,
)
from .compute_cost import accumulate_logical, grouped_breakdown, with_dependencies
from .errors import AnalysisError
from .facts import TARGET_MEMORY_OWNER, MemoryHierarchyFacts
from .footprint import (
    cached_level,
    footprint_of,
    reached_by,
    reuse_windows,
    wave_of,
)
from .iteration_scope import IterationScope, Repeats
from .liveness import analyze_liveness, result_copies
from .metadata import (
    Breakdown,
    Footprint,
    MemoryLevelPeak,
    MemoryMetadata,
    RegionMemoryMetadata,
    Spread,
    Traffic,
    TrafficBytes,
    ValueLifetime,
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
            scope.access.projected_relations(call, ctx)
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
                    logical_groups=(((), whole.get(name, TrafficBytes())),),
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

    groups: dict = field(default_factory=dict)
    total: dict[str, TrafficBytes] = field(default_factory=dict)
    per_unit: dict[str, dict[str, TrafficBytes]] = field(default_factory=dict)


def _account_shares(
    accounts: _TrafficAccounts, topologies: tuple[str, ...]
) -> Breakdown[TrafficBytes]:
    """Build a breakdown without encoding an account as a topology key."""
    names = sorted({*accounts.total, *accounts.per_unit})
    return grouped_breakdown(
        Breakdown(
            tuple(
                (
                    name,
                    Spread(
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
    ctx: MemoryContext,
) -> None:
    """Add one occurrence to each independent Function counting domain."""
    for into, stated in (
        (storage, record.traffic.storage),
        (communication, record.traffic.communication),
    ):
        for name, spread in stated.kinds:
            accumulate_logical(into.groups, name, spread, ctx)
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
    values: tuple[ValueLifetime, ...],
    region: MeshRegion,
    entered_at: int,
    exited_at: int,
    facts: MemoryHierarchyFacts,
    topology_names: tuple[str, ...],
    owners,
) -> tuple[ValueLifetime, ...]:
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
        lifetime = item
        if (
            not lifetime.persistent and lifetime.last_used_at < entered_at
        ) or lifetime.defined_at > exited_at:
            continue
        declared = facts.explicit(lifetime.memory_level)
        if declared is None or not declared.owner:
            raise AnalysisError(f"memory: level {lifetime.memory_level!r} has no declared owner")
        if declared.owner == TARGET_MEMORY_OWNER:
            continue
        try:
            private = owners.private(lifetime.memory_level, topology_names[outermost_region])
        except ValueError:
            raise AnalysisError(
                f"memory: level {lifetime.memory_level!r} owner {declared.owner!r} "
                f"is not a declared topology in {topology_names}"
            )
        held = leaves_of(lifetime.value.expr.type)[lifetime.value.index]
        if isinstance(held, TensorType) and is_unsharded_layout(held.layout):
            if not entered_at < lifetime.defined_at < exited_at:
                continue
        else:
            if not private:
                continue
            layout = shard_layout_of(getattr(held, "layout", None))
            if layout is None or not held_in_region(participant_layout(layout).mesh, region.mesh):
                continue
        result.append(item)
    return tuple(result)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    units: tuple[str, ...] = ()
    storage: _TrafficAccounts = field(default_factory=_TrafficAccounts)
    communication: _TrafficAccounts = field(default_factory=_TrafficAccounts)
    memory_level: str | None = None
    wave: tuple[int, int] | None = None


class MemoryVisitor(ValuationVisitor[None]):
    """Attach per-Call traffic and aggregate it over each enclosing loop."""

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: MemoryContext
    ) -> None:
        if not isinstance(expr, Call):
            return
        recorded = (SSAValue(expr),) in ctx.scope.access.accesses.get("narrow", {})
        if not ctx.units:
            raise AnalysisError("memory: visitor context is missing cost contexts")
        callee = ctx.callee
        if callee is None:
            moved = (
                call_traffic(
                    expr,
                    replace(ctx, topology_level=None),
                    {unit: replace(ctx, topology_level=unit) for unit in ctx.units},
                    ctx.scope,
                    ctx.topology_level,
                )
                if recorded
                else MemoryMetadata()
            )
            dependencies = ctx.scope.provenance.dependencies(expr)

            moved = replace(
                moved,
                traffic=Traffic(
                    with_dependencies(moved.traffic.storage, dependencies),
                    with_dependencies(moved.traffic.communication, dependencies),
                ),
            )
        else:
            summary = get_metadata(callee, RegionMemoryMetadata, ctx=ctx)
            if summary is None:
                raise AnalysisError(f"memory: missing summary for {callee.name!r}")
            moved = MemoryMetadata(topologies=summary.topologies, traffic=summary.traffic)
        attach(expr, moved)
        repeats = ctx.scope.repeats_of(expr, ctx.topology_level)
        add_traffic(
            ctx.storage,
            ctx.communication,
            moved,
            repeats,
            ctx,
        )


def analyze_memory(function: Function, context: AnalyzeContext) -> None:
    """Attach one definition's movement, reached addresses and liveness."""
    module = context.module
    topology_level = context.topology_level
    facts = context.target.get_facts(MemoryHierarchyFacts)
    owner_topologies = tuple(
        level.name for level in context.target.get_facts(TopologyFacts).topologies
    )
    for level in facts.explicit_levels:
        if not level.owner:
            raise AnalysisError(f"memory: level {level.name!r} has no declared owner")
        try:
            context.target.storage_owners
        except ValueError:
            raise AnalysisError(
                f"memory: level {level.name!r} owner {level.owner!r} is not a "
                f"target topology in {owner_topologies}"
            )
    topologies = context.topologies
    cache = cached_level(facts)
    wave = (
        wave_of(module, context.target, topology_level, topologies=topologies)
        if cache is not None
        else None
    )
    memory_level = cache[1] if cache is not None and wave is not None else None
    units = tuple(topology.name for topology in topologies) or (
        (topology_level,) if topology_level else ()
    )
    memory_context = MemoryContext(
        **vars(context),
        units=tuple(dict.fromkeys(units)),
        memory_level=memory_level,
        wave=wave,
    )
    MemoryVisitor().visit(function.body, memory_context)
    liveness = analyze_liveness(function, context, is_callee=function is not context.entry)
    attach(function, liveness)
    allocation = AllocationConstraintVisitor()
    attach(function, allocation.run(function, context))
    for value in allocation.by_value.values():
        expr = value.value.expr
        if id(expr) not in context.scope.snapshot.definition.locations:
            continue
        moved = get_metadata(expr, MemoryMetadata)
        if moved is not None:
            attach(expr, replace(moved, buffer_bytes=None, offsets=()))
    footprint = None
    if memory_level is not None and wave is not None:
        groups = defaultdict(list)
        available = True
        for path, (call, _accesses) in context.scope.root.access.accesses.get("narrow", {}).items():
            scope = context.scope.root.at(path[:1])
            moved = get_metadata(call, MemoryMetadata)
            origin, loops = scope._access_environment(path)
            evidence = reached_by(
                scope,
                call,
                memory_level=memory_level,
                wave_units=wave[0],
                declared_units=wave[1],
                operands=() if moved is None else moved.operands,
                ctx=replace(context, topology_level=None),
                window=len(scope.loop_scopes) - 1,
                path=path,
                loops=loops,
            )
            if evidence is None:
                available = False
                continue
            windows = -(-origin.mesh_positions(topology_level) // wave[0])
            trips = prod(loop.trips() for loop in loops)
            key = (path[:-1], origin, windows * trips)
            groups[key].extend(evidence)
            if len(path) == 1 and moved is not None:
                attach(call, replace(moved, footprint=footprint_of(evidence, windows=windows)))
        if available:
            footprint = Footprint(
                tuple(
                    (windows, tuple(items)) for (_path, _origin, windows), items in groups.items()
                )
            )
    reuse = (
        ()
        if memory_level is None or wave is None
        else reuse_windows(
            context.scope.root,
            memory_level=memory_level,
            wave=wave,
            ctx=replace(context, topology_level=topology_level),
            capacity=cache[2],
        )
    )
    peaks, lifetimes, status = (), (), "not_run"
    if function is context.entry:
        peaks, lifetimes, status = _place(allocation, facts, memory_context.units)
        occupied = {peak.memory_level for peak in peaks}
        peaks += tuple(
            MemoryLevelPeak(name, 0, 0, facts.explicit(name).capacity_bytes)
            for name in memory_context.storage.total
            if name not in occupied and facts.explicit(name) is not None
        )
        peaks = tuple(sorted(peaks, key=lambda peak: peak.memory_level))
    errors, advisories = _memory_findings(peaks, footprint, reuse, cache, wave)
    attach(
        function,
        RegionMemoryMetadata(
            topologies=memory_context.units,
            traffic=Traffic(
                storage=_account_shares(memory_context.storage, memory_context.units),
                communication=_account_shares(memory_context.communication, memory_context.units),
            ),
            footprint=footprint,
            peaks=peaks,
            lifetimes=lifetimes,
            solver_status=status,
            reuse_windows=reuse,
            errors=errors,
            advisories=advisories,
        ),
    )


def _resident_peaks(model, windows, offsets, facts, sizes, *, interval=None):
    """Count live root spans, using address unions only for solved arenas."""
    levels = {}
    for window in windows:
        value = model.by_value.get(window.value)
        if value is not None:
            levels.setdefault(window.memory_level, []).append((model.root(value), window))
    peaks = []
    for name, items in sorted(levels.items()):
        events = {
            point for _value, window in items for point in (window.defined_at, window.last_used_at)
        }
        if interval is not None:
            start, end = interval
            events = {start, end, *(p for p in events if start <= p <= end)}

        def occupied(point, persistent=False):
            live = {}
            for value, window in items:
                if window.persistent or (
                    not persistent and window.defined_at <= point <= window.last_used_at
                ):
                    size = sizes[value] if window.persistent else window.bytes
                    live[value] = max(live.get(value, 0), size)
            if name not in ("gmem", "smem"):
                return sum(live.values())
            end = amount = 0
            for start, stop in sorted(
                (offsets[value], offsets[value] + size) for value, size in live.items()
            ):
                amount += max(0, stop - max(start, end))
                end = max(end, stop)
            return amount

        level = facts.explicit(name)
        peaks.append(
            MemoryLevelPeak(
                name,
                max((occupied(p) for p in events), default=0),
                occupied(0, True),
                level.capacity_bytes if level else None,
            )
        )
    return tuple(peaks)


def _place(model, facts, topologies):
    """Solve the entry's flat windows once and publish shared definition addresses."""
    model._groups = model._occupancy = None
    options = (
        model.context.options
        if isinstance(model.context.options, MemoryOptions)
        else MemoryOptions()
    )
    offsets, high_water = {}, {}
    for name in ("gmem", "smem"):
        solved = solve_allocation(
            get_metadata(model.context.function, AllocationSummaryMetadata),
            model.context,
            options,
            memory_level=name,
        )
        offsets.update(solved.offsets)
        high_water[name] = solved.peak_bytes
    sizes = {value: model.size(value) for value in model.roots()}
    grouped = {}
    for value in model.by_value.values():
        if (
            value.lifetime.persistent
            or value.lifetime.memory_level not in high_water
            or any(
                alias.kind in (AliasKind.VIEW, AliasKind.FORWARD, AliasKind.INSERT_SLICE)
                for alias in value.aliases
            )
        ):
            continue
        grouped.setdefault(id(value.value.expr), (value.value.expr, []))[1].append(value)
    for expr, values in grouped.values():
        moved = get_metadata(expr, MemoryMetadata) or MemoryMetadata()
        values.sort(key=lambda value: value.value.index)
        addresses = tuple(
            offsets[model.root(value)] for value in values if not value.lifetime.persistent
        )
        copies = result_copies(expr)
        size = values[0].lifetime.bytes // copies
        if len(values) == 1 and addresses:
            addresses = tuple(addresses[0] + copy * size for copy in range(copies))
        attach(
            expr,
            replace(
                moved,
                offsets=addresses,
                buffer_bytes=None if isinstance(expr.type, TupleType) else size,
            ),
        )
    peaks = _resident_peaks(model, model.windows, offsets, facts, sizes)
    peaks = tuple(
        replace(peak, peak_bytes=high_water.get(peak.memory_level, peak.peak_bytes))
        for peak in peaks
    )
    regions, claimed = {}, set()
    rmem_peak = 0
    names = tuple(level.name for level in model.context.target.get_facts(TopologyFacts).topologies)
    for window in model.liveness.regions:
        resident = values_in_region(
            tuple(model.windows),
            window.region,
            window.entered_at,
            window.exited_at,
            facts,
            names,
            model.context.target.storage_owners,
        )
        levels = _resident_peaks(
            model, resident, offsets, facts, sizes, interval=(window.entered_at, window.exited_at)
        )
        if not levels:
            continue
        region, held = regions.setdefault(id(window.region), (window.region, {}))
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
            if level.memory_level == "rmem":
                rmem_peak = max(rmem_peak, level.peak_bytes)
                claimed.update(id(value) for value in resident if value.memory_level == "rmem")
    for region, held in regions.values():
        attach(
            region,
            RegionMemoryMetadata(
                solver_status="feasible", topologies=topologies, peaks=tuple(held.values())
            ),
        )
    if claimed:
        others = _resident_peaks(
            model, tuple(w for w in model.windows if id(w) not in claimed), offsets, facts, sizes
        )
        other = next((p.peak_bytes for p in others if p.memory_level == "rmem"), 0)
        peaks = tuple(
            replace(p, peak_bytes=max(rmem_peak, other)) if p.memory_level == "rmem" else p
            for p in peaks
        )
    lifetimes = []
    windows = defaultdict(list)
    for window in model.windows:
        windows[window.value].append(window)
    for value in model.by_value.values():
        if any(alias.kind in (AliasKind.VIEW, AliasKind.FORWARD) for alias in value.aliases):
            continue
        if id(value.value.expr) in model.context.scope.snapshot.definition.locations:
            lifetimes.append(value.lifetime)
        else:
            lifetimes.extend(windows[value.value])
    return peaks, tuple(lifetimes), "feasible"


def _memory_findings(peaks, footprint, reuse, cache, wave):
    """Build placement and cache findings from this definition's conclusions."""
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
            spread.logical
            for _value, _ownership, held in footprint.buffers
            for _level, spread in held.kinds
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


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
]
