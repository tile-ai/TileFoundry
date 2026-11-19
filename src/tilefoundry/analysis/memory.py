"""Memory-family projection from shared IterationScope and Access records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import prod

from tilefoundry import target, utils
from tilefoundry.interfaces import access, cost, valuation
from tilefoundry.ir import core, expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.hir import sharding

from .access import AliasKind
from .allocation import (
    AllocationConstraintVisitor,
    AllocationSummaryMetadata,
    held_on_one_unit,
    owner_units,
    solve_allocation,
)
from .compute_cost import accumulate_logical, grouped_breakdown, with_dependencies
from .errors import Error
from .footprint import (
    cached_level,
    reached_by,
    reuse_windows,
    wave_of,
)
from .iteration_scope import IterationScope, Repeats
from .liveness import analyze_liveness, result_copies
from .metadata import MemoryLevelPeak, MemoryMetadata, RegionMemoryMetadata, Traffic, ValueLifetime
from .precision import Precision
from .quantity import Breakdown, Spread
from .reach import Footprint, footprint_of
from .visitor import AnalyzeContext

SELECTOR = "memory"
UMAT_CONSUMPTION_LEVEL = str(expr.StorageKind.RMEM)


@dataclass(frozen=True)
class MemoryOptions:
    """How long the address assignment may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


def reached_bytes(
    boundaries: tuple[tuple[types.Type, object], ...], umat_level: str | None
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
    for held, acc in boundaries:
        leaves = ir_utils.leaves_of(held)
        if not leaves:
            return None
        if len(leaves) == 1:
            taken = {0: bytes_for(leaves[0], access.reached_elements(acc))}
        else:
            reached = access.reached_leaves(acc, len(leaves))
            if reached is None:
                return None
            taken = {index: access.static_bytes(leaves[index]) for index in sorted(reached)}
        for index, size in taken.items():
            if size is None:
                return None
            total += size
            leaf = leaves[index]
            memory_level = umat_level if leaf.storage is expr.StorageKind.UMAT else str(leaf.storage)
            if memory_level is not None:
                by_memory_level[memory_level] = by_memory_level.get(memory_level, 0) + size
    return total, by_memory_level


def bytes_for(held: types.Type, elements: int | None) -> int | None:
    """The bytes *elements* of *held* occupy, or ``None`` when unanswerable.

    Counted from how wide one element is, rounded up: a packed dtype has no
    whole number of bytes per element, so a share of the whole would round one
    bool to nothing and nine of them to one byte instead of two. A value whose
    element width nobody states is not one this can answer for.
    """
    if elements is None or not isinstance(held, types.TensorType):
        return None
    bits = getattr(held.dtype, "bit_width", None)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        return None
    return -(-elements * bits // 8)


def movement(
    call: expr.Call,
    cost_value: cost.Cost,
    ctx: cost.Context,
    type_list: tuple[types.Type, ...],
    scope: IterationScope | None = None,
) -> tuple[tuple[tuple[str, cost.TrafficBytes], ...], tuple[cost.TrafficBytes, ...]]:
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
    if len(cost_value.traffic) != len(operands):
        raise Error(
            f"{expr.describe_value(call)}: cost reports {len(cost_value.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    traffic = cost_value.traffic
    if isinstance(call.target, core.FunctionRef):
        charged = [
            ir_utils.bytes_by_storage(
                type_,
                umat_level=UMAT_CONSUMPTION_LEVEL if index < len(call.args) else None,
            )
            for index, type_ in enumerate(type_list)
        ]
    else:
        if not access.relation_registry.has(type(call.target)):
            raise Error(
                f"{expr.describe_value(call)}: states no access relations, so nothing here "
                "says what it moves"
            )
        local_relations = (
            scope.access.projected_relations(call, ctx)
            if scope is not None
            else access.projected(access.relations_of(call, ctx), call, ctx)
        )
        result = ctx.local_type_of(call)
        fields = ir_utils.result_fields(result)
        outputs = local_relations[len(call.args) :]
        if len(fields) > len(outputs):
            raise Error(
                f"{expr.describe_value(call)}: states {len(outputs)} output "
                f"boundaries for a result of {len(fields)} fields"
            )
        amounts, charged = [], []
        for index, moved in enumerate(cost_value.traffic):
            if index == len(call.args):
                asked = tuple((field_, outputs[position]) for position, field_ in enumerate(fields))
                memory_level = None
            else:
                asked = ((type_list[index], local_relations[index]),)
                memory_level = UMAT_CONSUMPTION_LEVEL
            answer = reached_bytes(asked, memory_level)
            if answer is None:
                raise Error(
                    f"{expr.describe_value(call)}: boundary {index} reaches coordinates "
                    "nothing here can charge in bytes"
                )
            moving, by_memory_level = answer
            amounts.append(cost.TrafficBytes(moving if moved.read else 0, moving if moved.write else 0))
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
    by_memory_level = (
        tuple(
            (name, cost.TrafficBytes(reads.get(name, 0), writes.get(name, 0)))
            for name in sorted(set(reads) | set(writes))
        )
        if moving
        else ()
    )
    return by_memory_level, traffic


def call_traffic(
    node: expr.Call,
    whole: cost.Context,
    locals_by_unit: "dict[str, cost.Context]",
    scope: IterationScope | None = None,
    counted: str | None = None,
) -> MemoryMetadata:
    """What one Call moves, whole and for one unit.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    The Type of the leaf it reached names the level those bytes are charged at,
    and an allocation does not correct either answer. A Reshard onto new units
    counts one copy per added unit.
    """
    storage_whole: dict[str, cost.TrafficBytes] = {}
    storage_per_unit: dict[str, dict[str, cost.TrafficBytes]] = {}
    crossing_whole: dict[str, cost.TrafficBytes] = {}
    crossing_per_unit: dict[str, dict[str, cost.TrafficBytes]] = {}
    operands: tuple[cost.TrafficBytes, ...] = ()
    for topology, ctx in ((None, whole), *locals_by_unit.items()):
        try:
            cost_value = cost.Visitor().visit(node, ctx)
        except (ValueError, expr.VerifyError) as error:
            raise Error(str(error)) from None
        type_list = (
            *(ctx.local_type_of(arg) for arg in node.args),
            ctx.local_type_of(node),
        )
        by_memory_level, positional = movement(node, cost_value, ctx, type_list, scope)
        for memory_level, moved in by_memory_level:
            if topology is None:
                storage_whole[memory_level] = moved
            else:
                storage_per_unit.setdefault(memory_level, {})[topology] = moved
        for boundary, moved in cost_value.sent:
            if topology is not None and finer_than(topology, boundary, locals_by_unit):
                continue
            if topology is None:
                crossing_whole[boundary] = moved
            else:
                crossing_per_unit.setdefault(boundary, {})[topology] = moved
        if topology is None:
            operands = positional
    added = (
        sharding.added_units(node, access.target_owners(whole.target))
        if isinstance(node.target, sharding.Reshard)
        else 0
    )
    units = (lambda _: added) if added else scope.mesh_units if scope is not None else lambda _: 1
    counted = counted or next(iter(locals_by_unit))
    return MemoryMetadata(
        topologies=tuple(locals_by_unit),
        traffic=Traffic(
            storage=occurrence_shares(
                storage_whole, storage_per_unit, tuple(locals_by_unit), counted, units
            ),
            communication=occurrence_shares(
                crossing_whole, crossing_per_unit, tuple(locals_by_unit), None, units
            ),
        ),
        operands=operands,
    )


def finer_than(topology: str, boundary: str, ordered: "dict[str, cost.Context]") -> bool:
    """Whether *topology* sits inside *boundary* rather than at or around it.

    Crossing a boundary is what the units on either side of it do. A unit
    inside one did not cross it, and stating a share for it would divide a
    move nobody made.
    """
    names = tuple(ordered)
    if topology not in names or boundary not in names:
        return False
    return names.index(topology) > names.index(boundary)


def occurrence_shares(
    whole: dict[str, cost.TrafficBytes],
    per_unit: dict[str, dict[str, cost.TrafficBytes]],
    topologies: tuple[str, ...],
    counted: str | None,
    units: Callable[[str | None], int],
) -> "Breakdown[cost.TrafficBytes]":
    """One entry per name, each carrying the whole and every topology's share.

    ``counted`` selects storage's unit; None counts each communication boundary
    at its own topology. Total scales that share by its executing units.
    The shares run in *topologies* order, so a topology a move never reached --
    a unit inside the boundary it crossed -- reads as no bytes rather than as
    a missing entry.
    """
    entries = []
    for name in sorted({*whole, *per_unit}):
        topology = counted if counted is not None else name
        if topology not in topologies:
            raise Error(f"memory: counted topology {topology!r} is not declared")
        moved = per_unit.get(name, {}).get(topology, cost.TrafficBytes())
        factor = units(topology)
        entries.append(
            (
                name,
                Spread(
                    logical_groups=(((), whole.get(name, cost.TrafficBytes())),),
                    total=cost.TrafficBytes(moved.read * factor, moved.write * factor),
                    per_unit=tuple(
                        per_unit.get(name, {}).get(each, cost.TrafficBytes()) for each in topologies
                    ),
                ),
            )
        )
    return Breakdown(tuple(entries))


@dataclass
class TrafficAccounts:
    """Function traffic kept separate in its three counting domains."""

    groups: dict = field(default_factory=dict)
    total: dict[str, cost.TrafficBytes] = field(default_factory=dict)
    per_unit: dict[str, dict[str, cost.TrafficBytes]] = field(default_factory=dict)


def account_shares(
    accounts: TrafficAccounts, topologies: tuple[str, ...]
) -> Breakdown[cost.TrafficBytes]:
    """Build a breakdown without encoding an account as a topology key."""
    names = sorted({*accounts.total, *accounts.per_unit})
    return grouped_breakdown(
        Breakdown(
            tuple(
                (
                    name,
                    Spread(
                        total=accounts.total.get(name, cost.TrafficBytes()),
                        per_unit=tuple(
                            accounts.per_unit.get(name, {}).get(topology, cost.TrafficBytes())
                            for topology in topologies
                        ),
                    ),
                )
                for name in names
            )
        ),
        accounts.groups,
    )


def accumulate_traffic(
    into: dict[str, cost.TrafficBytes], name: str, moved: cost.TrafficBytes, trips: int
) -> None:
    running = into.get(name, cost.TrafficBytes())
    into[name] = cost.TrafficBytes(
        running.read + moved.read * trips,
        running.write + moved.write * trips,
    )


def add_traffic(
    storage: TrafficAccounts,
    communication: TrafficAccounts,
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
            accumulate_traffic(into.total, name, spread.total, repeats.loop_trips)
            for topology, moved in zip(record.topologies, spread.per_unit, strict=False):
                accumulate_traffic(
                    into.per_unit.setdefault(name, {}), topology, moved, repeats.loop_trips
                )


def held_in_mesh_scope(mesh: types.Mesh, mesh_scope: types.Mesh) -> bool:
    """Whether *mesh_scope* selects holders of *mesh* at every topology it names.

    *mesh* may also name enclosing topologies. Those do not reject a mesh scope
    that states only an inner topology; each topology of *mesh_scope* is compared
    independently, by the unit ids of its ``utils.topology_mesh``.
    """
    for name in ir_utils.topology_names(mesh_scope):
        part = ir_utils.topology_mesh(mesh, name)
        if part is None or not ir_utils.within_mesh_scope(
            part, ir_utils.topology_mesh(mesh_scope, name)
        ):
            return False
    return True


def values_in_region(
    values: tuple[ValueLifetime, ...],
    region: core.MeshRegion,
    entered_at: int,
    exited_at: int,
    facts: target.MemoryHierarchyFacts,
    topology_names: tuple[str, ...],
    owners,
) -> tuple[ValueLifetime, ...]:
    """Resident values whose lifetimes and ownership intersect one mesh region.

    Unsharded tensor leaves defined in the window contribute while live,
    including definitions in child regions.
    """
    owner_order = {name: index for index, name in enumerate(topology_names)}
    region_indices = tuple(
        owner_order[name] for name in ir_utils.topology_names(region.mesh)
    )
    outermost_region = min(region_indices)
    result = []
    for item in values:
        lifetime = item
        if (
            not lifetime.persistent and lifetime.last_used_at < entered_at
        ) or lifetime.defined_at > exited_at:
            continue
        declared = facts.explicit(lifetime.memory_level)
        if declared is None or not declared.owner:
            raise Error(f"memory: level {lifetime.memory_level!r} has no declared owner")
        if declared.owner == expr.TARGET_MEMORY_OWNER:
            continue
        try:
            private = owners.private(lifetime.memory_level, topology_names[outermost_region])
        except ValueError:
            raise Error(
                f"memory: level {lifetime.memory_level!r} owner {declared.owner!r} "
                f"is not a declared topology in {topology_names}"
            )
        held = ir_utils.leaves_of(lifetime.value.expr.type)[lifetime.value.index]
        if isinstance(held, types.TensorType) and ir_utils.is_unsharded_layout(held.layout):
            if not entered_at < lifetime.defined_at < exited_at:
                continue
        else:
            if not private:
                continue
            layout = types.shard_layout_of(getattr(held, "layout", None))
            layout = (
                types.shard_layout_of(ir_utils.physical_layout(layout))
                if layout is not None
                else None
            )
            if layout is None or not held_in_mesh_scope(ir_utils.filter_topologies(layout).mesh, region.mesh):
                continue
        result.append(item)
    return tuple(result)


@dataclass
class MemoryContext(AnalyzeContext):
    """State carried through the memory-family expression walk."""

    topology_names: tuple[str, ...] = ()
    storage: TrafficAccounts = field(default_factory=TrafficAccounts)
    communication: TrafficAccounts = field(default_factory=TrafficAccounts)
    memory_level: str | None = None
    wave: tuple[int, int] | None = None


class MemoryVisitor(valuation.Visitor[None]):
    """Attach per-Call traffic and aggregate it over each enclosing loop."""

    def default_visit_leaf(
        self, node: expr.Expr, operands: tuple[None, ...], ctx: MemoryContext
    ) -> None:
        if not isinstance(node, expr.Call):
            return
        recorded = (core.SSAValue(node),) in ctx.scope.access.accesses.get("narrow", {})
        if not ctx.topology_names:
            raise Error("memory: visitor context is missing cost contexts")
        callee = ctx.callee
        if callee is None:
            moved = (
                call_traffic(
                    node,
                    replace(ctx, topology=None),
                    {topology: replace(ctx, topology=topology) for topology in ctx.topology_names},
                    ctx.scope,
                    ctx.topology,
                )
                if recorded
                else MemoryMetadata()
            )
            dependencies = ctx.scope.provenance.dependencies(node)

            moved = replace(
                moved,
                traffic=Traffic(
                    with_dependencies(moved.traffic.storage, dependencies),
                    with_dependencies(moved.traffic.communication, dependencies),
                ),
            )
        else:
            summary = expr.get_metadata(callee, RegionMemoryMetadata, ctx=ctx)
            if summary is None:
                raise Error(f"memory: missing summary for {callee.name!r}")
            moved = MemoryMetadata(topologies=summary.topologies, traffic=summary.traffic)
        node.attach_metadata(moved)
        repeats = ctx.scope.repeats_of(node, ctx.topology)
        add_traffic(
            ctx.storage,
            ctx.communication,
            moved,
            repeats,
            ctx,
        )


def analyze_memory(function: core.Function, context: AnalyzeContext) -> None:
    """Attach one definition's movement, reached addresses and liveness."""
    module = context.module
    topology = context.topology
    facts = context.target.get_facts(target.MemoryHierarchyFacts)
    owner_topologies = tuple(
        limits.name for limits in context.target.get_facts(target.TopologyFacts).topologies
    )
    for explicit in facts.explicit_levels:
        if not explicit.owner:
            raise Error(f"memory: level {explicit.name!r} has no declared owner")
        try:
            context.target.storage_owners
        except ValueError:
            raise Error(
                f"memory: level {explicit.name!r} owner {explicit.owner!r} is not a "
                f"target topology in {owner_topologies}"
            )
    topologies = context.topologies
    cache = cached_level(facts)
    wave = (
        wave_of(module, context.target, topology, topologies=topologies)
        if cache is not None
        else None
    )
    memory_level = cache[1] if cache is not None and wave is not None else None
    topology_names = tuple(each.name for each in topologies) or (
        (topology,) if topology else ()
    )
    memory_context = MemoryContext(
        **vars(context),
        topology_names=tuple(dict.fromkeys(topology_names)),
        memory_level=memory_level,
        wave=wave,
    )
    MemoryVisitor().visit(function.body, memory_context)
    liveness = analyze_liveness(function, context, is_callee=function is not context.entry)
    function.attach_metadata(liveness)
    allocation = AllocationConstraintVisitor()
    function.attach_metadata(allocation.run(function, context))
    for value in allocation.by_value.values():
        node = value.value.expr
        if id(node) not in context.scope.snapshot.definition.locations:
            continue
        moved = expr.get_metadata(node, MemoryMetadata)
        if moved is not None:
            node.attach_metadata(replace(moved, buffer_bytes=None, offsets=()))
    footprint = None
    if memory_level is not None and wave is not None:
        groups = defaultdict(list)
        available = True
        for path, (call, _) in context.scope.root.access.accesses.get("narrow", {}).items():
            scope = context.scope.root.at(path[:1])
            moved = expr.get_metadata(call, MemoryMetadata)
            origin, loops = scope.access_environment(path)
            evidence = reached_by(
                scope,
                call,
                memory_level=memory_level,
                wave_units=wave[0],
                declared_units=wave[1],
                operands=() if moved is None else moved.operands,
                ctx=replace(context, topology=None),
                window=len(scope.loop_scopes) - 1,
                path=path,
                loops=loops,
            )
            if evidence is None:
                available = False
                continue
            windows = -(-origin.mesh_units(topology) // wave[0])
            trips = prod(loop.trips() for loop in loops)
            key = (path[:-1], origin, windows * trips)
            groups[key].extend(evidence)
            if len(path) == 1 and moved is not None:
                call.attach_metadata(replace(moved, footprint=footprint_of(evidence, windows=windows)))
        if available:
            footprint = Footprint(
                tuple((windows, tuple(items)) for (_, _, windows), items in groups.items())
            )
    reuse = (
        ()
        if memory_level is None or wave is None
        else reuse_windows(
            context.scope.root,
            memory_level=memory_level,
            wave=wave,
            ctx=replace(context, topology=topology),
            capacity=cache[2],
        )
    )
    peaks, lifetimes, status = (), (), "not_run"
    if function is context.entry:
        peaks, lifetimes, status = place(allocation, facts, memory_context.topology_names)
        occupied = {peak.memory_level for peak in peaks}
        peaks += tuple(
            MemoryLevelPeak(name, 0, 0, facts.explicit(name).capacity_bytes)
            for name in memory_context.storage.total
            if name not in occupied and facts.explicit(name) is not None
        )
        peaks = tuple(sorted(peaks, key=lambda peak: peak.memory_level))
    errors, advisories = memory_findings(peaks, footprint, reuse, cache, wave)
    function.attach_metadata(
        RegionMemoryMetadata(
            topologies=memory_context.topology_names,
            traffic=Traffic(
                storage=account_shares(memory_context.storage, memory_context.topology_names),
                communication=account_shares(memory_context.communication, memory_context.topology_names),
            ),
            footprint=footprint,
            peaks=peaks,
            lifetimes=lifetimes,
            solver_status=status,
            reuse_windows=reuse,
            errors=errors,
            advisories=advisories,
        ))


def resident_peaks(model, windows, offsets, facts, sizes, *, interval=None):
    """Count live root spans, using address unions only for solved arenas."""
    by_memory_level = {}
    for window in windows:
        value = model.by_value.get(window.value)
        if value is not None:
            by_memory_level.setdefault(window.memory_level, []).append((model.root(value), window))
    owners = model.context.target.storage_owners
    units = {
        value: owner_units(value.lifetime, owners)
        for items in by_memory_level.values()
        for value, _ in items
    }
    peaks = []
    for name, items in sorted(by_memory_level.items()):
        events = {
            point for _, window in items for point in (window.defined_at, window.last_used_at)
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
                return held_on_one_unit(live, units)
            end = amount = 0
            for start, stop in sorted(
                (offsets[value], offsets[value] + size) for value, size in live.items()
            ):
                amount += max(0, stop - max(start, end))
                end = max(end, stop)
            return amount

        explicit = facts.explicit(name)
        peaks.append(
            MemoryLevelPeak(
                name,
                max((occupied(p) for p in events), default=0),
                occupied(0, True),
                explicit.capacity_bytes if explicit else None,
            )
        )
    return tuple(peaks)


def place(model, facts, topologies):
    """Solve the entry's flat windows once and publish shared definition addresses."""
    model.forget_indexes()
    options = (
        model.context.options
        if isinstance(model.context.options, MemoryOptions)
        else MemoryOptions()
    )
    offsets, high_water = {}, {}
    for name in ("gmem", "smem"):
        solved = solve_allocation(
            expr.get_metadata(model.context.function, AllocationSummaryMetadata),
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
    for node, values in grouped.values():
        moved = expr.get_metadata(node, MemoryMetadata) or MemoryMetadata()
        values.sort(key=lambda value: value.value.index)
        addresses = tuple(
            offsets[model.root(value)] for value in values if not value.lifetime.persistent
        )
        copies = result_copies(node)
        size = values[0].lifetime.bytes // copies
        if len(values) == 1 and addresses:
            addresses = tuple(addresses[0] + copy * size for copy in range(copies))
        node.attach_metadata(replace(
                moved,
                offsets=addresses,
                buffer_bytes=None if isinstance(node.type, types.TupleType) else size,
            )
        )
    peaks = resident_peaks(model, model.windows, offsets, facts, sizes)
    peaks = tuple(
        replace(peak, peak_bytes=high_water.get(peak.memory_level, peak.peak_bytes))
        for peak in peaks
    )
    regions, claimed = {}, set()
    rmem_peak = 0
    names = tuple(limits.name for limits in model.context.target.get_facts(target.TopologyFacts).topologies)
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
        peaks_here = resident_peaks(
            model, resident, offsets, facts, sizes, interval=(window.entered_at, window.exited_at)
        )
        if not peaks_here:
            continue
        region, held = regions.setdefault(id(window.region), (window.region, {}))
        for peak in peaks_here:
            previous = held.get(peak.memory_level)
            held[peak.memory_level] = (
                peak
                if previous is None
                else replace(
                    peak,
                    peak_bytes=max(previous.peak_bytes, peak.peak_bytes),
                    persistent_bytes=max(previous.persistent_bytes, peak.persistent_bytes),
                )
            )
            if peak.memory_level == "rmem":
                rmem_peak = max(rmem_peak, peak.peak_bytes)
                claimed.update(id(value) for value in resident if value.memory_level == "rmem")
    for region, held in regions.values():
        region.attach_metadata(RegionMemoryMetadata(
                solver_status="feasible", topologies=topologies, peaks=tuple(held.values())
            ))
    if claimed:
        others = resident_peaks(
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


def memory_findings(peaks, footprint, reuse, cache, wave):
    """Build address and cache findings from this definition's conclusions."""
    errors = tuple(
        f"{peak.memory_level} peak {utils.format_bytes(peak.peak_bytes)} exceeds "
        f"capacity {utils.format_bytes(peak.capacity_bytes)}"
        for peak in peaks
        if peak.exceeds_capacity
    )
    advisories = ()
    if cache is not None and wave is not None and footprint is not None:
        reliable = (Precision.EXACT, Precision.LOWER_BOUND)
        messages = {}
        for row in reuse:
            if not row.fits:
                messages.setdefault(row.time or row.space, (row.holds_bytes, row.precision))
        for window, (held, precision) in messages.items():
            message = (
                f"{cache[0]} reuse window {window} holds {utils.format_bytes(held)} at a "
                f"{wave[0]}-unit wave, exceeding capacity {utils.format_bytes(cache[2])}"
            )
            if precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
        used = sum(spread.logical for _, _, held in footprint.buffers for _, spread in held.kinds)
        if used >= cache[2] and not any(
            not row.fits
            and not row.time
            and row.holds_bytes == used
            and (row.precision in reliable) == (footprint.precision in reliable)
            for row in reuse
        ):
            message = (
                f"{cache[0]} working set {utils.format_bytes(used)} at the first iteration of a "
                f"{wave[0]}-unit wave exceeds capacity {utils.format_bytes(cache[2])}"
            )
            if footprint.precision in reliable:
                errors += (message,)
            else:
                advisories += (message,)
    return errors, advisories


def owned_bytes(summary: AllocationSummaryMetadata, leaf: core.SSAValue) -> tuple[int, int] | None:
    """The shared memory bytes *leaf*'s allocation owns, as the memory family placed them.

    Every stage's offset with the stage size, all stages together; a Tuple
    result records no stage size, so the leaf's own size stands for it. None
    when the allocation has no placed offset.
    """
    root = summary.root(leaf)
    placed = expr.get_metadata(root.value.expr, MemoryMetadata)
    offsets = () if placed is None else placed.offsets
    if not offsets:
        return None
    size = root.lifetime.bytes if placed.buffer_bytes is None else placed.buffer_bytes
    return min(offsets), max(offsets) + size


__all__ = [
    "MemoryOptions",
    "SELECTOR",
    "analyze_memory",
    "bytes_for",
    "call_traffic",
    "owned_bytes",
    "reached_bytes",
]
