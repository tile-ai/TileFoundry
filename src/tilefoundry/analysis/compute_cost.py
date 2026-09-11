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
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Mesh, composed, topology_axes
from tilefoundry.ir.types.shard.mesh import _positions_layout
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.visitor_registry.contexts import (
    CostContext,
    FunctionScope,
    TrafficBytes,
)
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import PerformanceServiceFacts, ThroughputFacts
from .metadata import Breakdown, ComputeCostMetadata, breakdown, shares
from .visitor import AnalyzeContext

SELECTOR = "compute-cost"


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    moved: "TrafficMetadata | None" = None,
    *,
    unit: str,
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
        all(not value for _name, value in shares(cost.flops, cost.topologies, unit).items())
        and all(not value for _kind, value in shares(cost.service, cost.topologies, unit).items())
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
    topology_level: str,
    scale: int = 1,
) -> int:
    """Price one occurrence's projected work against one unit's throughputs.

    Compute, movement and what leaves the unit overlap, so the duration is
    whichever takes longest; the same bytes can spend two resources and still
    take one span of time. Work with no stated throughput is refused rather
    than priced at nothing. Movement at a level the target publishes no
    bandwidth for is a different case -- stated and left untimed, because a
    rate nobody published is not one this may invent.
    """
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
    for name, share in shares(cost.flops, cost.topologies, topology_level).items():
        value = share
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

    for kind, share in shares(cost.service, cost.topologies, topology_level).items():
        value = share
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


def _named(flops: dict) -> dict[str, int]:
    """The flop counts keyed by dtype name, as a record states them."""
    return {dtype.name: value for dtype, value in flops.items()}


def _call_cost_record(
    expr: Call,
    locals_by_unit: "dict[str, CostContext]",
    positions_by_unit: "dict[str, int]",
    asked: "str | None" = None,
) -> ComputeCostMetadata:
    """Measure the work one Call asks for, without attaching the record.

    Work only: what an occurrence moves is the memory family's answer. One
    unit's work is measured at every declared level rather than at one chosen
    for the reader, and global work is any of them repeated over the positions
    executing this scope -- the product is the same whichever level states it.
    """
    flops_by_unit: list[dict[str, int]] = []
    service_by_unit: list[dict[str, int]] = []
    whole_flops: dict[str, int] = {}
    whole_service: dict[str, int] = {}
    for unit, local in locals_by_unit.items():
        try:
            cost = CostEvaluator().visit(expr, local)
        except (ValueError, VerifyError) as error:
            raise AnalysisError(str(error)) from None
        flops_by_unit.append(_named(cost.flops))
        service_by_unit.append(dict(cost.service))
        if unit == (asked or next(iter(locals_by_unit))):
            repeats = positions_by_unit[unit]
            whole_flops = {name: value * repeats for name, value in _named(cost.flops).items()}
            whole_service = {kind: value * repeats for kind, value in cost.service.items()}
    return ComputeCostMetadata(
        topologies=tuple(locals_by_unit),
        flops=breakdown(whole_flops, flops_by_unit, 0),
        service=breakdown(whole_service, service_by_unit, 0),
    )


def _scope_position_count(mesh: Mesh, topology_level: str | None, topologies: tuple) -> int:
    """Count positions at or above the selected level within *mesh*."""
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
    held: "Breakdown[TrafficBytes]",
    topologies: tuple[str, ...],
    kind: str,
    topology_level: "str | None",
) -> int:
    """One kind's bytes for one unit of *topology_level*, read and written together."""
    moved = shares(held, topologies, topology_level).get(kind)
    return moved.total_bytes if moved is not None else 0


def _accumulate(
    flops: dict[str, int],
    service: dict[str, int],
    by_unit: "dict[str, dict[str, dict[str, int]]]",
    record: ComputeCostMetadata,
    trips: int,
) -> None:
    for kind, spread in record.flops.kinds:
        flops[kind] = flops.get(kind, 0) + spread.total * trips
    for kind, spread in record.service.kinds:
        service[kind] = service.get(kind, 0) + spread.total * trips
    for index, unit in enumerate(record.topologies):
        held = by_unit.setdefault(unit, {"flops": {}, "service": {}})
        for kind, spread in record.flops.kinds:
            held["flops"][kind] = held["flops"].get(kind, 0) + spread.at(index) * trips
        for kind, spread in record.service.kinds:
            held["service"][kind] = held["service"].get(kind, 0) + spread.at(index) * trips


@dataclass
class ComputeCostContext(AnalyzeContext):
    """State carried through the compute-cost expression walk.

    ``executing_positions`` is how many positions of each declared level the
    enclosing scope holds; it is the multiplier that turns one unit's work at
    that level into total replicated work.
    """

    locals_by_unit: dict[str, CostContext] = field(default_factory=dict)
    current_mesh: Mesh | None = None
    executing_positions: dict[str, int] = field(default_factory=dict)
    flops: dict[str, int] = field(default_factory=dict)
    service: dict[str, int] = field(default_factory=dict)
    by_unit: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    call_count: list[int] = field(default_factory=lambda: [0])


class ComputeCostVisitor(ExprVisitor[None]):
    """Attach per-Call work and accumulate multiplicity-aware totals."""

    def visit_MeshRegion(self, expr: MeshRegion, ctx: ComputeCostContext) -> None:
        """Carry the region's execution multiplicity into each contained Call."""
        for arg in expr.args:
            self.visit(arg, ctx)
        mesh = composed((ctx.current_mesh, expr.mesh)) if ctx.current_mesh else expr.mesh
        topologies = ctx.module.effective_topologies()
        positions = {
            unit: _scope_position_count(mesh, unit, topologies) for unit in ctx.locals_by_unit
        }
        self.visit(
            expr.body,
            replace(ctx, executing_positions=positions, current_mesh=mesh),
        )

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
        if not ctx.locals_by_unit:
            raise AnalysisError("compute-cost: visitor context is missing its cost context")
        record = _call_cost_record(
            expr, ctx.locals_by_unit, ctx.executing_positions, ctx.topology_level
        )
        attach(expr, record)
        owner = ctx.current if id(expr) in ctx.current.accesses["narrow"] else ctx.root
        repeats = 1
        cursor = owner
        while cursor.parent is not None:
            if cursor.is_variant(expr):
                repeats *= max(1, cursor.trips())
            cursor = cursor.parent
        _accumulate(ctx.flops, ctx.service, ctx.by_unit, record, repeats)


def analyze_compute_cost(
    function: Function,
    context: AnalyzeContext,
) -> None:
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
        executing_positions=dict.fromkeys(locals_by_unit, 1),
    )
    ComputeCostVisitor().visit(function.body, cost_context)
    if cost_context.call_count[0] > 0:
        attach(
            function,
            ComputeCostMetadata(
                topologies=tuple(cost_context.by_unit),
                flops=breakdown(
                    dict(cost_context.flops),
                    [dict(held["flops"]) for held in cost_context.by_unit.values()],
                    0,
                ),
                service=breakdown(
                    dict(cost_context.service),
                    [dict(held["service"]) for held in cost_context.by_unit.values()],
                    0,
                ),
            ),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
