"""How much work the authored program asks for.

This family reads the program and nothing else. Flops come from each op's
registered cost evaluator and bytes from the logical types its operands and
result carry, so the record it leaves is the same on every backend. What that
work costs in time is a separate question, asked by the roofline family against
a target's rates.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, Type
from tilefoundry.target import Target
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import ThroughputFacts
from .metadata import ComputeCostMetadata, TrafficBytes
from .walk import (
    attach,
    bytes_by_storage,
    describe,
    enclosing_trips,
    postorder,
    reachable_functions,
)

SELECTOR = "compute-cost"


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    facts: ThroughputFacts,
) -> bool:
    """Whether an occurrence has no work priced by the target's local rates."""
    return all(not value for _name, value in cost.flops_per_unit) and not (
        cost.traffic_per_unit_at(facts.bandwidth_level).total_bytes
    )


def _local_duration_ns(
    cost: ComputeCostMetadata,
    facts: ThroughputFacts,
    *,
    level: str,
    scale: int = 1,
) -> int:
    """Price one occurrence's projected work against one unit's rates."""
    if facts.rate_unit != level:
        raise AnalysisError(
            f"timeline: selected topology level {level!r}, but the target's "
            f"per-unit rates are stated for {facts.rate_unit!r}"
        )

    if _is_structural_occurrence(cost, facts):
        return 0

    compute_ns = 0
    for name, value in cost.flops_per_unit:
        if not value:
            continue
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise AnalysisError(f"timeline: unknown compute dtype {name!r}")
        rate = facts.peak_per_unit_for(dtype)
        if rate is None or rate <= 0:
            raise AnalysisError(
                f"timeline: target publishes no per-unit compute rate for "
                f"dtype {name!r} at {level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // rate)

    traffic = cost.traffic_per_unit_at(facts.bandwidth_level)
    moved = traffic.total_bytes * scale
    if not moved:
        unmodelled = tuple(
            name
            for name, value in cost.traffic_per_unit
            if name != facts.bandwidth_level and value.total_bytes
        )
        if unmodelled:
            names = ", ".join(repr(name) for name in unmodelled)
            raise AnalysisError(
                f"timeline: occurrence traffic is only at unmodelled storage "
                f"level(s) {names}; target bandwidth is stated for "
                f"{facts.bandwidth_level!r}"
            )

    memory_ns = 0
    if moved:
        rate = facts.memory_bandwidth_bytes_per_second_per_unit
        if rate is None or rate <= 0:
            raise AnalysisError(
                f"timeline: target publishes no per-unit bandwidth for level "
                f"{facts.bandwidth_level!r} at {level!r}"
            )
        memory_ns = -(-(moved * 1_000_000_000) // rate)
    return max(compute_ns, memory_ns)


def _call_movement(
    call: Call,
    cost: Cost,
    operand_types: tuple[Type, ...] | None = None,
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and where those bytes are charged.

    How much moves is the op's answer; which level it moves at is a function of
    that operand's Type. *operand_types* supplies the projected types for the
    per-unit reading.
    """
    operands = (*call.args, call)
    types = operand_types or tuple(operand.type for operand in operands)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    charged: list[TrafficBytes] = []
    if len(types) != len(operands):  # pragma: no cover - internal caller contract
        raise AnalysisError("cost movement needs one projected type per operand")
    for type_, moved in zip(types, cost.traffic):
        by_level = bytes_by_storage(type_)
        charged.append(moved)
        if len(by_level) == 1:
            (level,) = by_level
            reads[level] = reads.get(level, 0) + moved.read
            writes[level] = writes.get(level, 0) + moved.write
            continue


        for level, value in by_level.items():
            if moved.read:
                reads[level] = reads.get(level, 0) + value
            if moved.write:
                writes[level] = writes.get(level, 0) + value
    levels = (
        ()
        if cost.bytes == 0
        else tuple(
            (level, TrafficBytes(reads.get(level, 0), writes.get(level, 0)))
            for level in sorted(set(reads) | set(writes))
        )
    )
    return levels, tuple(charged)


def _flops(flops: dict) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((dtype.name, value) for dtype, value in flops.items()))


def _call_cost_record(
    expr: Call,
    whole: CostEvaluator,
    local: CostEvaluator,
) -> ComputeCostMetadata:
    """Measure one Call without attaching the resulting record."""
    try:
        whole_cost = whole.visit(expr)
        local_cost = local.visit(expr)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    traffic_by_level, operands = _call_movement(expr, whole_cost)
    local_types = (
        *(local.ctx.local_type_of(arg) for arg in expr.args),
        local.ctx.local_output_type(expr),
    )
    local_traffic, _local_operands = _call_movement(expr, local_cost, local_types)
    return ComputeCostMetadata(
        flops=_flops(whole_cost.flops),
        flops_per_unit=_flops(local_cost.flops),
        traffic=traffic_by_level,
        traffic_per_unit=local_traffic,
        operands=operands,
    )


def _accumulate(
    flops: dict[str, int],
    flops_per_unit: dict[str, int],
    traffic: dict[str, TrafficBytes],
    traffic_per_unit: dict[str, TrafficBytes],
    record: ComputeCostMetadata,
    trips: int,
) -> None:
    for name, value in record.flops:
        flops[name] = flops.get(name, 0) + value * trips
    for name, value in record.flops_per_unit:
        flops_per_unit[name] = flops_per_unit.get(name, 0) + value * trips
    for level, value in record.traffic:
        current = traffic.get(level, TrafficBytes())
        traffic[level] = TrafficBytes(
            current.read + value.read * trips,
            current.write + value.write * trips,
        )
    for level, value in record.traffic_per_unit:
        current = traffic_per_unit.get(level, TrafficBytes())
        traffic_per_unit[level] = TrafficBytes(
            current.read + value.read * trips,
            current.write + value.write * trips,
        )


def analyze_compute_cost(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach one-trip work per Call and multiplicity-aware totals per Function."""
    topologies = module.effective_topologies()
    for fn in reachable_functions(function):
        scope = FunctionScope(module, fn)
        whole = CostEvaluator(CostContext(scope=scope))
        local = CostEvaluator(
            CostContext(scope=scope, level=level, topologies=topologies)
        )
        flops: dict[str, int] = {}
        flops_per_unit: dict[str, int] = {}
        traffic: dict[str, TrafficBytes] = {}
        traffic_per_unit: dict[str, TrafficBytes] = {}
        trips = enclosing_trips(fn.body)
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            count = trips.get(id(expr), 1)
            record = _call_cost_record(expr, whole, local)
            attach(expr, record)
            _accumulate(
                flops,
                flops_per_unit,
                traffic,
                traffic_per_unit,
                record,
                count,
            )
        attach(
            fn,
            ComputeCostMetadata(
                flops=tuple(sorted(flops.items())),
                flops_per_unit=tuple(sorted(flops_per_unit.items())),
                traffic=tuple(sorted(traffic.items())),
                traffic_per_unit=tuple(sorted(traffic_per_unit.items())),
            ),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
