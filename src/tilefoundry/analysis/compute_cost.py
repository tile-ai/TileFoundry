"""How much work the authored program asks for.

This family reads the program and nothing else. Flops and typed service come
from each op's registered cost evaluator, so the record it leaves is the same on
every backend. What that work moves is the memory family's half of the same
declaration, and what it costs in time is a separate question again, asked
against a target's rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Expr, VerifyError
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.mesh_scope import MeshScope
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Mesh, composed, level_axes
from tilefoundry.ir.types.shard.mesh import _positions_layout
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.contexts import (
    CostContext,
    FunctionScope,
)
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .metadata import ComputeCostMetadata
from .visitor import AnalyzeContext

SELECTOR = "compute-cost"


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    moved: "TrafficMetadata | None" = None,
    *,
    bandwidth_level: str | None = None,
) -> bool:
    """Whether an occurrence asks for nothing this model puts on a clock.

    Only what could take time is counted: the flops, the typed service, and the
    bytes at the one level a bandwidth is published for. Movement at any other
    level is still movement and still recorded -- what it is not is work this
    model can lay on a timeline, so it neither earns a duration nor asks for a
    placement to be laid at. Having moved bytes and having timed work are
    different questions, and this is the second one.
    """
    return (
        all(not value for _name, value in cost.flops_per_unit)
        and all(not value for _kind, value in cost.service_per_unit)
        and not (
            moved.per_unit_at(bandwidth_level).total_bytes
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
    level: str,
    scale: int = 1,
) -> int:
    """Price one occurrence's projected work against one unit's throughputs.

    Compute and movement overlap within one occurrence, so its duration is
    whichever side takes longer. Work with no stated throughput is refused
    rather than priced at nothing: a number with a hole in it reads as a program
    that does less than it does. Movement at a level the target publishes no
    bandwidth for is a different case -- it is stated and left untimed, because
    a rate nobody published is not one this may invent.
    """
    if services.unit != level:
        raise AnalysisError(
            f"performance: selected topology level {level!r}, but the target's "
            f"one-unit throughputs are stated for {services.unit!r}"
        )

    if _is_structural_occurrence(cost, moved, bandwidth_level=facts.bandwidth_level):
        return 0

    compute_ns = 0
    for name, value in cost.flops_per_unit:
        if not value:
            continue
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise AnalysisError(f"performance: unknown compute dtype {name!r}")
        throughput = services.flops(dtype)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for "
                f"dtype {name!r} at {level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    for kind, value in cost.service_per_unit:
        if not value:
            continue
        throughput = services.ops(kind)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for "
                f"{kind!r} work at {level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // throughput)

    crossed = (
        moved.per_unit_at(facts.bandwidth_level).total_bytes * scale
        if moved is not None
        else 0
    )
    memory_ns = 0
    if crossed:
        throughput = services.bandwidth(facts.bandwidth_level)
        if throughput is None or throughput <= 0:
            raise AnalysisError(
                f"performance: target states no one-unit throughput for level "
                f"{facts.bandwidth_level!r} at {level!r}"
            )
        memory_ns = -(-(crossed * 1_000_000_000) // throughput)
    return max(compute_ns, memory_ns)


def _flops(flops: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((dtype.name, value) for dtype, value in flops.items()))


def _call_cost_record(
    expr: Call, local: CostContext, executing_positions: int
) -> ComputeCostMetadata:
    """Measure the work one Call asks for, without attaching the record.

    Work only: what an occurrence moves is the memory family's answer. Global
    work is one selected unit's work repeated over the positions executing this
    scope, so it is derived from the same registered evaluator rather than
    independently recomputed from the authored type.
    """
    try:
        local_cost = CostEvaluator().visit(expr, local)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    return ComputeCostMetadata(
        flops=_flops(
            {dtype: value * executing_positions for dtype, value in local_cost.flops.items()}
        ),
        flops_per_unit=_flops(local_cost.flops),
        service=tuple(
            sorted(
                (kind, value * executing_positions)
                for kind, value in local_cost.service.items()
            )
        ),
        service_per_unit=tuple(sorted(local_cost.service.items())),
    )


def _scope_position_count(
    mesh: Mesh, level: str | None, topologies: tuple
) -> int:
    """Count positions at or above the selected level within *mesh*."""
    if level is None:
        return 1
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    selected = declared[level]
    shape, _strides, _offset = _positions_layout(mesh)
    positions = 1
    for topology, axes in zip(mesh.topologies, level_axes(mesh)):
        if declared[topology.name] > selected:
            continue
        for axis in axes:
            extent = shape[axis]
            if not isinstance(extent, int) or isinstance(extent, bool) or extent < 1:
                raise AnalysisError(
                    f"compute-cost: mesh axis {axis} needs a positive static extent, "
                    f"got {extent!r}"
                )
            positions *= extent
    return positions


def _accumulate(
    flops: dict[str, int],
    flops_per_unit: dict[str, int],
    service: dict[str, int],
    service_per_unit: dict[str, int],
    record: ComputeCostMetadata,
    trips: int,
) -> None:
    for name, value in record.flops:
        flops[name] = flops.get(name, 0) + value * trips
    for name, value in record.flops_per_unit:
        flops_per_unit[name] = flops_per_unit.get(name, 0) + value * trips
    for name, value in record.service:
        service[name] = service.get(name, 0) + value * trips
    for name, value in record.service_per_unit:
        service_per_unit[name] = service_per_unit.get(name, 0) + value * trips


@dataclass
class ComputeCostContext(AnalyzeContext):
    """State carried through the compute-cost expression walk.

    ``executing_positions`` is the number of positions in the enclosing scope
    through the selected topology level; it is the D68 multiplier that turns
    per-unit work into total replicated work.
    """

    local: CostContext | None = None
    current_mesh: Mesh | None = None
    executing_positions: int = 1
    flops: dict[str, int] = field(default_factory=dict)
    flops_per_unit: dict[str, int] = field(default_factory=dict)
    service: dict[str, int] = field(default_factory=dict)
    service_per_unit: dict[str, int] = field(default_factory=dict)
    call_count: list[int] = field(default_factory=lambda: [0])


class ComputeCostVisitor(ExprVisitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def visit_MeshScope(self, expr: MeshScope, ctx: ComputeCostContext) -> None:
        """Carry the region's execution multiplicity into each contained Call."""
        for arg in expr.args:
            self.visit(arg, ctx)
        mesh = composed((ctx.current_mesh, expr.mesh)) if ctx.current_mesh else expr.mesh
        positions = _scope_position_count(
            mesh, ctx.level, ctx.module.effective_topologies()
        )
        self.visit(
            expr.body,
            replace(ctx, executing_positions=positions, current_mesh=mesh),
        )

    def visit_GridRegionExpr(
        self, expr: GridRegionExpr, ctx: ComputeCostContext
    ) -> None:
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
        if ctx.local is None:
            raise AnalysisError("compute-cost: visitor context is missing its cost context")
        record = _call_cost_record(expr, ctx.local, ctx.executing_positions)
        attach(expr, record)
        owner = ctx.current if id(expr) in ctx.current.accesses["narrow"] else ctx.root
        repeats = 1
        cursor = owner
        while cursor.parent is not None:
            if cursor.is_variant(expr):
                repeats *= max(1, cursor.trips())
            cursor = cursor.parent
        _accumulate(
            ctx.flops,
            ctx.flops_per_unit,
            ctx.service,
            ctx.service_per_unit,
            record,
            repeats,
        )


def analyze_compute_cost(
    function: Function,
    context: AnalyzeContext,
) -> None:
    """Attach one-trip work per Call and multiplicity-aware totals per Function."""
    module, level = context.module, context.level
    topologies = module.effective_topologies()
    scope = FunctionScope(module, function)
    local = CostContext(scope=scope, level=level, topologies=topologies)
    cost_context = ComputeCostContext(
        module=module,
        target=context.target,
        level=level,
        options=context.options,
        root=context.root,
        current=context.current,
        local=local,
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    if cost_context.call_count[0] > 0:
        attach(
            function,
            ComputeCostMetadata(
                flops=tuple(sorted(cost_context.flops.items())),
                flops_per_unit=tuple(sorted(cost_context.flops_per_unit.items())),
                service=tuple(sorted(cost_context.service.items())),
                service_per_unit=tuple(sorted(cost_context.service_per_unit.items())),
            ),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
