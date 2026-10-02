"""Count typed floating-point and other operations in authored HIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, VerifyError
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import DType
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .iteration_scope import Repeats
from .metadata import Breakdown, ComputeCostMetadata, MemoryMetadata, breakdown, shares
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


def _accumulate(ctx: "ComputeCostContext", record: ComputeCostMetadata, repeats: Repeats) -> None:
    _add(ctx.flops_logical, _domain_values(record.flops, "logical"), repeats.varying_loop_trips)
    _add(ctx.flops, _domain_values(record.flops, "total"), repeats.loop_trips)
    _add(
        ctx.other_ops_logical,
        _domain_values(record.other_ops, "logical"),
        repeats.varying_loop_trips,
    )
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
    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    whole: CostContext | None = None
    flops: dict[str, int] = field(default_factory=dict)
    flops_logical: dict[str, int] = field(default_factory=dict)
    other_ops: dict[str, int] = field(default_factory=dict)
    other_ops_logical: dict[str, int] = field(default_factory=dict)
    by_unit: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    call_count: list[int] = field(default_factory=lambda: [0])
    precision: list[AnalysisPrecision] = field(default_factory=lambda: [AnalysisPrecision.EXACT])


class ComputeCostVisitor(ExprVisitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def visit_MeshRegion(self, expr: MeshRegion, ctx: ComputeCostContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        for arg in expr.args:
            self.visit(arg, ctx)
        self.visit(expr.body, replace(ctx, current=child))

    def visit_LoopRegion(self, expr: LoopRegion, ctx: ComputeCostContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.args:
            self.visit(operand, ctx)
        self.visit(expr.body, inner)
        for operand in expr.yield_values:
            self.visit(operand, inner)

    def default_visit_leaf(
        self, expr: Expr, _operands: tuple[None, ...], ctx: ComputeCostContext
    ) -> None:
        if not isinstance(expr, Call):
            return
        ctx.call_count[0] += 1
        if not ctx.locals_by_unit or ctx.whole is None:
            raise AnalysisError("compute-cost: visitor context is missing its cost context")
        repeats = ctx.current.repeats_of(expr, ctx.topology_level)
        record = _call_cost_record(
            expr, ctx.locals_by_unit, repeats.mesh_positions, ctx.whole, ctx.topology_level
        )
        attach(expr, record)
        ctx.precision[0] = ctx.precision[0].join(repeats.trips_precision)
        _accumulate(ctx, record, repeats)


def analyze_compute_cost(function: Function, context: AnalyzeContext) -> None:
    """Attach one-trip work per Call and multiplicity-aware totals per Function."""
    module, topology_level = context.module, context.topology_level
    topologies = module.effective_topologies()
    scope = FunctionScope(module, function)
    units = tuple(topology.name for topology in topologies) or (topology_level,)
    locals_by_unit = {
        unit: CostContext(scope=scope, topology_level=unit, topologies=topologies)
        for unit in units
        if unit is not None
    }
    cost_context = ComputeCostContext(
        module=module,
        target=context.target,
        topology_level=topology_level,
        options=context.options,
        root=context.root,
        current=context.current,
        locals_by_unit=locals_by_unit,
        whole=CostContext(scope=scope),
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    if cost_context.call_count[0] > 0:
        units = tuple(cost_context.by_unit)
        attach(
            function,
            ComputeCostMetadata(
                topologies=units,
                precision=cost_context.precision[0],
                flops=breakdown(
                    cost_context.flops,
                    tuple(cost_context.by_unit[u]["flops"] for u in units),
                    0,
                    logical=cost_context.flops_logical,
                ),
                other_ops=breakdown(
                    cost_context.other_ops,
                    tuple(cost_context.by_unit[u]["other_ops"] for u in units),
                    0,
                    logical=cost_context.other_ops_logical,
                ),
            ),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
