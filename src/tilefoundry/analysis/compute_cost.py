"""Count typed floating-point and other operations in authored HIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry import target
from tilefoundry.interfaces import cost, valuation
from tilefoundry.ir import core, expr

from .errors import Error
from .iteration_scope import Repeats
from .liveness import result_copies
from .metadata import ComputeCostMetadata, MemoryMetadata
from .precision import Precision
from .quantity import Breakdown, added, breakdown, scaled, shares
from .visitor import AnalyzeContext

SELECTOR = "compute-cost"


def counts(values: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((getattr(kind, "name", kind), value) for kind, value in values.items()))


def level_value(held: Breakdown[int], topologies: tuple[str, ...], level: str | None):
    return tuple(shares(held, topologies, level).items())


def is_structural_occurrence(
    cost_value: ComputeCostMetadata,
    moved: MemoryMetadata | None = None,
    *,
    unit: str,
) -> bool:
    """Whether an occurrence asks for nothing this model puts on a clock."""
    return (
        all(not value for _, value in level_value(cost_value.flops, cost_value.topologies, unit))
        and all(
            not value for _, value in level_value(cost_value.other_ops, cost_value.topologies, unit)
        )
        and not any(
            traffic.total_bytes
            for traffic in (
                shares(moved.traffic.storage, moved.topologies, unit).values()
                if moved is not None
                else ()
            )
        )
    )


def local_duration_ns(
    cost_value: ComputeCostMetadata,
    facts: target.ThroughputFacts,
    services: target.PerformanceServiceFacts,
    *,
    moved: MemoryMetadata | None = None,
    topology_level: str | None = None,
    level: str | None = None,
    scale: int = 1,
    positions: int = 1,
) -> int:
    """Price one occurrence's projected work against one unit's throughputs.

    At the bandwidth level, one unit moves its share of the occurrence's
    footprint: one window's bytes over the *positions* that window holds.
    """
    topology_level = topology_level if topology_level is not None else level
    if topology_level is None:
        raise Error("performance: no topology level was selected")
    if services.unit != topology_level:
        raise Error(
            f"performance: selected topology level {topology_level!r}, but the "
            f"target's one-unit throughputs are stated for {services.unit!r}"
        )
    if is_structural_occurrence(cost_value, moved, unit=topology_level):
        return 0

    compute_ns = 0
    for name, value in level_value(cost_value.flops, cost_value.topologies, topology_level):
        if not value:
            continue
        dtype = getattr(expr.DType, name, None)
        if dtype is None:
            raise Error(f"performance: unknown compute dtype {name!r}")
        throughput = services.flops(dtype)
        if throughput is None or throughput <= 0:
            raise Error(
                f"performance: target states no one-unit throughput for dtype "
                f"{name!r} at {topology_level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    for kind, value in level_value(cost_value.other_ops, cost_value.topologies, topology_level):
        if not value:
            continue
        throughput = services.ops(kind)
        if throughput is None or throughput <= 0:
            raise Error(
                f"performance: target states no one-unit throughput for {kind!r} "
                f"work at {topology_level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    memory_ns = 0
    storage_traffic = (
        shares(moved.traffic.storage, moved.topologies, topology_level) if moved is not None else {}
    )
    unique = None if moved is None or moved.footprint is None else moved.footprint.bytes_at(facts.bandwidth_level)
    for memory_level, traffic in storage_traffic.items():
        crossed = traffic.total_bytes * scale
        if unique is not None and memory_level == facts.bandwidth_level:
            crossed = -(-unique[0] // positions) * scale
        throughput = services.bandwidth(memory_level)
        if not crossed or throughput is None:
            continue
        if throughput <= 0:
            raise Error(
                f"performance: target states no one-unit throughput for level "
                f"{memory_level!r} at {topology_level!r}"
            )
        duration = -(-(crossed * 1_000_000_000) // throughput)
        memory_ns = max(memory_ns, duration)

    sent = (
        traffic_bytes(
            moved.traffic.communication,
            moved.topologies,
            topology_level,
            topology_level,
        )
        * scale
        if moved is not None
        else 0
    )
    link_ns = 0
    if sent:
        rate = services.bandwidth(topology_level)
        if rate:
            link_ns = -(-(sent * 1_000_000_000) // rate)
    return max(compute_ns, memory_ns, link_ns)


def inflight_bytes(call: expr.Call, scope, tgt, level: str) -> int | None:
    """Bytes one parallel unit keeps in flight for *call*'s traffic at *level*.

    An asynchronous copy (TMA, ``cp.async``) keeps every stage of its buffer in
    flight: one unit's bytes at *level* times the slots its buffer declares. A
    synchronous access keeps one 16-byte access per thread of the unit, and as
    many threads as the finest level allows when the program states none.
    """
    facts = tgt.get_facts(target.TopologyFacts)
    unit = facts.parallel()
    if unit is None or not facts.topologies:
        return None
    op = getattr(call.target, "op", call.target)
    if getattr(expr.schema_of(type(op)), "category", None) == "async":
        moved = expr.get_metadata(call, MemoryMetadata)
        stage = None if moved is None else shares(moved.traffic.storage, moved.topologies, unit.name).get(level)
        slots = max((result_copies(scope.capture_root(arg)) for arg in call.args), default=1)
        return None if stage is None else stage.total_bytes * slots
    finest = facts.topologies[-1]
    threads = finest.max_logical_units
    if scope.mesh is not None and finest.name in {item.name for item in scope.mesh.topologies}:
        threads = min(threads, scope.mesh_positions(finest.name) // scope.mesh_positions(unit.name))
    return None if threads is None else threads * 16


def call_cost_record(
    node: expr.Call,
    locals_by_unit: dict[str, cost.Context],
    positions: int,
    whole: cost.Context,
    asked: str | None = None,
) -> ComputeCostMetadata:
    """Measure one Call in logical, expanded, and every topology-unit domain."""
    flops_by_unit: list[tuple[tuple[str, int], ...]] = []
    other_ops_by_unit: list[tuple[tuple[str, int], ...]] = []
    total_flops: tuple[tuple[str, int], ...] = ()
    total_other_ops: tuple[tuple[str, int], ...] = ()
    try:
        logical = cost.Visitor().visit(node, whole)
    except (ValueError, expr.VerifyError) as error:
        raise Error(str(error)) from None
    for unit, local in locals_by_unit.items():
        try:
            cost_value = cost.Visitor().visit(node, local)
        except (ValueError, expr.VerifyError) as error:
            raise Error(str(error)) from None
        unit_flops = counts(cost_value.flops)
        unit_other_ops = counts(cost_value.service)
        flops_by_unit.append(unit_flops)
        other_ops_by_unit.append(unit_other_ops)
        if unit == (asked or next(iter(locals_by_unit))):
            total_flops = tuple((name, value * positions) for name, value in unit_flops)
            total_other_ops = tuple((kind, value * positions) for kind, value in unit_other_ops)
    return ComputeCostMetadata(
        topologies=tuple(locals_by_unit),
        flops=breakdown(
            dict(total_flops),
            tuple(dict(values) for values in flops_by_unit),
            0,
            logical=dict(counts(logical.flops)),
        ),
        other_ops=breakdown(
            dict(total_other_ops),
            tuple(dict(values) for values in other_ops_by_unit),
            0,
            logical=dict(counts(logical.service)),
        ),
    )


def traffic_bytes(
    held: Breakdown[cost.TrafficBytes],
    topologies: tuple[str, ...],
    kind: str,
    topology_level: str | None,
) -> int:
    moved = shares(held, topologies, topology_level).get(kind)
    return moved.total_bytes if moved is not None else 0


def add_counts(tgt: dict[str, int], values, trips: int) -> None:
    for name, value in values:
        tgt[name] = tgt.get(name, 0) + value * trips


def domain_values(held: Breakdown[int], domain: str) -> tuple[tuple[str, int], ...]:
    return tuple((name, getattr(spread, domain)) for name, spread in held.kinds)


def accumulate_logical(into, name, spread, ctx):
    """Consume local loop dependencies, retaining only this definition's formals."""
    precision = Precision.EXACT
    scope = (
        ctx.callee_scope
        if ctx.call_site is not None and ctx.callee_context() is not None
        else ctx.scope
    )
    for dependencies, value in spread.logical_groups:
        changing = set().union(
            *(scope.provenance.varies(dependency) for dependency in dependencies)
        )
        factor, bounded = scope.dependency_trips(dependencies)
        precision = precision.join(bounded)
        formals = tuple(formal for formal in ctx.scope.root.owner.params if id(formal) in changing)
        key = (name, tuple(id(formal) for formal in formals))
        value = scaled(value, factor)
        if key in into:
            value = added(into[key][1], value)
        into[key] = formals, value
    return precision


def with_dependencies(held, dependencies):
    """State a primitive's logical groups using its operand dependencies."""
    return replace(
        held,
        kinds=tuple(
            (name, replace(spread, logical_groups=((dependencies, spread.logical),)))
            for name, spread in held.kinds
        ),
    )


def grouped_breakdown(held, groups):
    """Publish a definition's logical groups beside its local numeric view."""
    return replace(
        held,
        kinds=tuple(
            (
                name,
                replace(
                    spread,
                    logical_groups=tuple(
                        (dependencies, value)
                        for (kind, _), (dependencies, value) in groups.items()
                        if kind == name
                    ),
                ),
            )
            for name, spread in held.kinds
        ),
    )


def accumulate(
    ctx: "ComputeCostContext",
    record: ComputeCostMetadata,
    repeats: Repeats,
) -> None:
    for held, groups in (
        (record.flops, ctx.flop_groups),
        (record.other_ops, ctx.other_groups),
    ):
        for name, spread in held.kinds:
            bounded = accumulate_logical(groups, name, spread, ctx)
            ctx.widen(bounded)
    add_counts(ctx.flops, domain_values(record.flops, "total"), repeats.loop_trips)
    add_counts(ctx.other_ops, domain_values(record.other_ops, "total"), repeats.loop_trips)
    for index, unit in enumerate(record.topologies):
        held = ctx.by_unit.setdefault(unit, {"flops": {}, "other_ops": {}})
        add_counts(
            held["flops"],
            ((name, spread.at(index)) for name, spread in record.flops.kinds),
            repeats.loop_trips,
        )
        add_counts(
            held["other_ops"],
            ((name, spread.at(index)) for name, spread in record.other_ops.kinds),
            repeats.loop_trips,
        )


@dataclass
class ComputeCostContext(AnalyzeContext):
    flop_groups: dict = field(default_factory=dict)
    other_groups: dict = field(default_factory=dict)
    units: tuple[str, ...] = ()
    flops: dict[str, int] = field(default_factory=dict)
    other_ops: dict[str, int] = field(default_factory=dict)
    by_unit: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    precision: list[Precision] = field(default_factory=lambda: [Precision.EXACT])

    def widen(self, precision: Precision) -> None:
        """Fold *precision* into the one every counted quantity so far has."""
        self.precision[0] = self.precision[0].join(precision)


class ComputeCostVisitor(valuation.Visitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def default_visit_leaf(
        self, node: expr.Expr, operands: tuple[None, ...], ctx: ComputeCostContext
    ) -> None:
        if not isinstance(node, expr.Call):
            return
        if not ctx.units:
            raise Error("compute-cost: visitor context is missing its cost context")
        repeats = ctx.scope.repeats_of(node, ctx.topology_level)
        callee = ctx.callee
        if callee is None:
            record = call_cost_record(
                node,
                {unit: replace(ctx, topology_level=unit) for unit in ctx.units},
                repeats.mesh_positions,
                replace(ctx, topology_level=None),
                ctx.topology_level,
            )
            dependencies = ctx.scope.provenance.dependencies(node)

            record = replace(
                record,
                flops=with_dependencies(record.flops, dependencies),
                other_ops=with_dependencies(record.other_ops, dependencies),
            )
        else:
            record = expr.get_metadata(callee, ComputeCostMetadata, ctx=ctx)
        node.attach_metadata(record)
        ctx.widen(repeats.trips_precision)
        ctx.widen(record.precision)
        accumulate(ctx, record, repeats)


def analyze_compute_cost(function: core.Function, context: AnalyzeContext) -> None:
    """Attach one-trip work per Call and multiplicity-aware totals per Function."""
    topology_level = context.topology_level
    topologies = context.topologies
    units = tuple(topology.name for topology in topologies) or (topology_level,)
    cost_context = ComputeCostContext(
        **vars(context), units=tuple(dict.fromkeys(unit for unit in units if unit is not None))
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    function.attach_metadata(summary(cost_context))


def summary(ctx):
    units = tuple(ctx.by_unit)
    return ComputeCostMetadata(
        topologies=units,
        precision=ctx.precision[0],
        flops=grouped_breakdown(
            breakdown(
                ctx.flops,
                tuple(ctx.by_unit[u]["flops"] for u in units),
                0,
            ),
            ctx.flop_groups,
        ),
        other_ops=grouped_breakdown(
            breakdown(
                ctx.other_ops,
                tuple(ctx.by_unit[u]["other_ops"] for u in units),
                0,
            ),
            ctx.other_groups,
        ),
    )


__all__ = [
    "SELECTOR",
    "accumulate_logical",
    "analyze_compute_cost",
    "grouped_breakdown",
    "inflight_bytes",
    "local_duration_ns",
    "with_dependencies",
]
