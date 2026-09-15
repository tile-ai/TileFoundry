"""Count typed floating-point and other operations in authored HIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, VerifyError
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Mesh, composed, topology_axes
from tilefoundry.ir.types.shard.mesh import _positions_layout
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .metadata import Breakdown, ComputeCostMetadata, breakdown, shares
from .visitor import AnalyzeContext

SELECTOR = "compute-cost"


def _counts(values: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((getattr(kind, "name", kind), value) for kind, value in values.items()))


def _at(held: Breakdown[int], topologies: tuple[str, ...], level: str | None):
    return tuple(shares(held, topologies, level).items())


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    moved: "TrafficMetadata | None" = None,
    *,
    unit: str,
    bandwidth_level: str | None = None,
) -> bool:
    """Whether an occurrence asks for nothing this model puts on a clock."""
    return (
        all(not value for _name, value in _at(cost.flops, cost.topologies, unit))
        and all(not value for _kind, value in _at(cost.other_ops, cost.topologies, unit))
        and not (
            _bytes(moved.storage, moved.topologies, bandwidth_level, unit)
            if moved is not None and bandwidth_level is not None
            else 0
        )
    )


def _local_duration_ns(
    cost: ComputeCostMetadata,
    facts: ThroughputFacts,
    services: PerformanceServiceFacts,
    *,
    moved: "TrafficMetadata | None" = None,
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
    if _is_structural_occurrence(
        cost, moved, unit=topology_level, bandwidth_level=facts.bandwidth_level
    ):
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

    crossed = (
        _bytes(moved.storage, moved.topologies, facts.bandwidth_level, topology_level) * scale
        if moved is not None
        else 0
    )
    memory_ns = 0
    if crossed:
        throughput = services.bandwidth(facts.bandwidth_level)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for level "
                f"{facts.bandwidth_level!r} at {topology_level!r}"
            )
        memory_ns = -(-(crossed * 1_000_000_000) // throughput)

    sent = (
        _bytes(moved.communication, moved.topologies, topology_level, topology_level) * scale
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
    positions_by_unit: dict[str, int],
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
            repeats = positions_by_unit[unit]
            total_flops = tuple((name, value * repeats) for name, value in unit_flops)
            total_other_ops = tuple((kind, value * repeats) for kind, value in unit_other_ops)
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


def _scope_position_count(mesh: Mesh, topology_level: str | None, topologies: tuple) -> int:
    if topology_level is None:
        return 1
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    selected = declared[topology_level]
    shape, _strides, _offset = _positions_layout(mesh)
    positions = 1
    for topology, axes in zip(mesh.topologies, topology_axes(mesh)):
        if declared[topology.name] > selected:
            continue
        for axis in axes:
            extent = shape[axis]
            if not isinstance(extent, int) or isinstance(extent, bool) or extent < 1:
                raise AnalysisError(
                    f"compute-cost: mesh axis {axis} needs a positive static extent, got {extent!r}"
                )
            positions *= extent
    return positions


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


def _accumulate(ctx: "ComputeCostContext", record: ComputeCostMetadata, trips: int) -> None:
    _add(ctx.flops_logical, _domain_values(record.flops, "logical"), trips)
    _add(ctx.flops, _domain_values(record.flops, "total"), trips)
    _add(ctx.other_ops_logical, _domain_values(record.other_ops, "logical"), trips)
    _add(ctx.other_ops, _domain_values(record.other_ops, "total"), trips)
    for index, unit in enumerate(record.topologies):
        held = ctx.by_unit.setdefault(unit, {"flops": {}, "other_ops": {}})
        _add(held["flops"], ((name, spread.at(index)) for name, spread in record.flops.kinds), trips)
        _add(
            held["other_ops"],
            ((name, spread.at(index)) for name, spread in record.other_ops.kinds),
            trips,
        )


@dataclass
class ComputeCostContext(AnalyzeContext):
    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    whole: CostContext | None = None
    current_mesh: Mesh | None = None
    executing_positions: dict[str, int] = field(default_factory=dict)
    flops: dict[str, int] = field(default_factory=dict)
    flops_logical: dict[str, int] = field(default_factory=dict)
    other_ops: dict[str, int] = field(default_factory=dict)
    other_ops_logical: dict[str, int] = field(default_factory=dict)
    by_unit: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    call_count: list[int] = field(default_factory=lambda: [0])


class ComputeCostVisitor(ExprVisitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def visit_MeshRegion(self, expr: MeshRegion, ctx: ComputeCostContext) -> None:
        for arg in expr.args:
            self.visit(arg, ctx)
        mesh = composed((ctx.current_mesh, expr.mesh)) if ctx.current_mesh else expr.mesh
        topologies = ctx.module.effective_topologies()
        positions = {
            unit: _scope_position_count(mesh, unit, topologies) for unit in ctx.locals_by_unit
        }
        self.visit(expr.body, replace(ctx, executing_positions=positions, current_mesh=mesh))

    def visit_LoopRegion(self, expr: LoopRegion, ctx: ComputeCostContext) -> None:
        child = next(item for item in ctx.current.children if item.owner is expr)
        inner = replace(ctx, current=child)
        for operand in expr.init_args:
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
        record = _call_cost_record(
            expr, ctx.locals_by_unit, ctx.executing_positions, ctx.whole, ctx.topology_level
        )
        attach(expr, record)
        owner = ctx.current if id(expr) in ctx.current.accesses["narrow"] else ctx.root
        repeats = 1
        cursor = owner
        while cursor.parent is not None:
            if cursor.is_variant(expr):
                repeats *= max(1, cursor.trips())
            cursor = cursor.parent
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
        executing_positions=dict.fromkeys(locals_by_unit, 1),
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    if cost_context.call_count[0] > 0:
        units = tuple(cost_context.by_unit)
        attach(
            function,
            ComputeCostMetadata(
                topologies=units,
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
