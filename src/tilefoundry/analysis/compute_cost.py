"""Count typed floating-point and other operations in authored HIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, VerifyError, get_metadata
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes
from tilefoundry.visitor_registry.valuation import ValuationVisitor
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .iteration_scope import Repeats
from .metadata import (
    Breakdown,
    ComputeCostMetadata,
    MemoryMetadata,
    added,
    breakdown,
    scaled,
    shares,
)
from .precision import AnalysisPrecision
from .visitor import AnalyzeContext

SELECTOR = "compute-cost"


def _counts(values: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((getattr(kind, "name", kind), value) for kind, value in values.items()))


def _at(held: Breakdown[int], topologies: tuple[str, ...], level: str | None):
    return tuple(shares(held, topologies, level).items())


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    moved: MemoryMetadata | None = None,
    *,
    unit: str,
) -> bool:
    """Whether an occurrence asks for nothing this model puts on a clock."""
    return (
        all(not value for _name, value in _at(cost.flops, cost.topologies, unit))
        and all(not value for _kind, value in _at(cost.other_ops, cost.topologies, unit))
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
    cost: ComputeCostMetadata,
    facts: ThroughputFacts,
    services: PerformanceServiceFacts,
    *,
    moved: MemoryMetadata | None = None,
    topology_level: str | None = None,
    level: str | None = None,
    scale: int = 1,
) -> int:
    """Price one occurrence's projected work against one unit's throughputs."""
    topology_level = topology_level if topology_level is not None else level
    if topology_level is None:
        raise AnalysisError("performance: no topology level was selected")
    if services.unit != topology_level:
        raise AnalysisError(
            f"performance: selected topology level {topology_level!r}, but the "
            f"target's one-unit throughputs are stated for {services.unit!r}"
        )
    if _is_structural_occurrence(cost, moved, unit=topology_level):
        return 0

    compute_ns = 0
    for name, value in _at(cost.flops, cost.topologies, topology_level):
        if not value:
            continue
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise AnalysisError(f"performance: unknown compute dtype {name!r}")
        throughput = services.flops(dtype)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for dtype "
                f"{name!r} at {topology_level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    for kind, value in _at(cost.other_ops, cost.topologies, topology_level):
        if not value:
            continue
        throughput = services.ops(kind)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for {kind!r} "
                f"work at {topology_level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    memory_ns = 0
    storage_traffic = (
        shares(moved.traffic.storage, moved.topologies, topology_level) if moved is not None else {}
    )
    for memory_level, traffic in storage_traffic.items():
        crossed = traffic.total_bytes * scale
        throughput = services.bandwidth(memory_level)
        if not crossed or throughput is None:
            continue
        if throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for level "
                f"{memory_level!r} at {topology_level!r}"
            )
        duration = -(-(crossed * 1_000_000_000) // throughput)
        memory_ns = max(memory_ns, duration)

    sent = (
        _bytes(
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


def _call_cost_record(
    expr: Call,
    locals_by_unit: dict[str, CostContext],
    positions: int,
    whole: CostContext,
    asked: str | None = None,
) -> ComputeCostMetadata:
    """Measure one Call in logical, expanded, and every topology-unit domain."""
    flops_by_unit: list[tuple[tuple[str, int], ...]] = []
    other_ops_by_unit: list[tuple[tuple[str, int], ...]] = []
    total_flops: tuple[tuple[str, int], ...] = ()
    total_other_ops: tuple[tuple[str, int], ...] = ()
    try:
        logical = CostEvaluator().visit(expr, whole)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    for unit, local in locals_by_unit.items():
        try:
            cost = CostEvaluator().visit(expr, local)
        except (ValueError, VerifyError) as error:
            raise AnalysisError(str(error)) from None
        unit_flops = _counts(cost.flops)
        unit_other_ops = _counts(cost.service)
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
            logical=dict(_counts(logical.flops)),
        ),
        other_ops=breakdown(
            dict(total_other_ops),
            tuple(dict(values) for values in other_ops_by_unit),
            0,
            logical=dict(_counts(logical.service)),
        ),
    )


def _bytes(
    held: Breakdown[TrafficBytes],
    topologies: tuple[str, ...],
    kind: str,
    topology_level: str | None,
) -> int:
    moved = shares(held, topologies, topology_level).get(kind)
    return moved.total_bytes if moved is not None else 0


def _add(target: dict[str, int], values, trips: int) -> None:
    for name, value in values:
        target[name] = target.get(name, 0) + value * trips


def _domain_values(held: Breakdown[int], domain: str) -> tuple[tuple[str, int], ...]:
    return tuple((name, getattr(spread, domain)) for name, spread in held.kinds)


def accumulate_logical(into, name, spread, ctx):
    """Consume local loop dependencies, retaining only this definition's formals."""
    precision = AnalysisPrecision.EXACT
    scope = (
        ctx.callee_scope
        if ctx.call_site is not None and ctx.callee_frame() is not None
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
                        for (kind, _ids), (dependencies, value) in groups.items()
                        if kind == name
                    ),
                ),
            )
            for name, spread in held.kinds
        ),
    )


def _accumulate(
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
            ctx.precision[0] = ctx.precision[0].join(bounded)
    _add(ctx.flops, _domain_values(record.flops, "total"), repeats.loop_trips)
    _add(ctx.other_ops, _domain_values(record.other_ops, "total"), repeats.loop_trips)
    for index, unit in enumerate(record.topologies):
        held = ctx.by_unit.setdefault(unit, {"flops": {}, "other_ops": {}})
        _add(
            held["flops"],
            ((name, spread.at(index)) for name, spread in record.flops.kinds),
            repeats.loop_trips,
        )
        _add(
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
    precision: list[AnalysisPrecision] = field(default_factory=lambda: [AnalysisPrecision.EXACT])


class ComputeCostVisitor(ValuationVisitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: ComputeCostContext
    ) -> None:
        if not isinstance(expr, Call):
            return
        if not ctx.units:
            raise AnalysisError("compute-cost: visitor context is missing its cost context")
        repeats = ctx.scope.repeats_of(expr, ctx.topology_level)
        callee_frame = ctx.callee_frame(expr)
        if callee_frame is None:
            record = _call_cost_record(
                expr,
                {unit: replace(ctx, topology_level=unit) for unit in ctx.units},
                repeats.mesh_positions,
                replace(ctx, topology_level=None),
                ctx.topology_level,
            )
            dependencies = ctx.scope.provenance.dependencies(expr)

            record = replace(
                record,
                flops=with_dependencies(record.flops, dependencies),
                other_ops=with_dependencies(record.other_ops, dependencies),
            )
        else:
            record = get_metadata(callee_frame.function, ComputeCostMetadata).resolve(ctx)
        attach(expr, record)
        ctx.precision[0] = ctx.precision[0].join(repeats.trips_precision).join(record.precision)
        _accumulate(ctx, record, repeats)


def analyze_compute_cost(function: Function, context: AnalyzeContext) -> None:
    """Attach one-trip work per Call and multiplicity-aware totals per Function."""
    topology_level = context.topology_level
    topologies = context.topologies
    units = tuple(topology.name for topology in topologies) or (topology_level,)
    cost_context = ComputeCostContext(
        **vars(context), units=tuple(dict.fromkeys(unit for unit in units if unit is not None))
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    attach(function, _summary(cost_context))


def _summary(ctx):
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


__all__ = ["SELECTOR", "analyze_compute_cost"]
