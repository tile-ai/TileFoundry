"""How much work the authored program asks for.

This family reads the program and nothing else. Flops come from each op's
registered cost evaluator and bytes from the logical types its operands and
result carry, so the record it leaves is the same on every backend. What that
work costs in time is a separate question, asked by the roofline family against
a target's rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import Target
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
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


@dataclass(frozen=True)
class _Totals:
    """One function's work, summed over the calls in its body."""

    flops: tuple[tuple[str, int], ...]
    flops_per_unit: tuple[tuple[str, int], ...]
    traffic: tuple[tuple[str, TrafficBytes], ...]


def _call_movement(
    call: Call, cost: Cost
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and where those bytes are charged.

    How much moves is the op's answer; which level it moves at is a function of
    that operand's Type.
    """
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    charged: list[TrafficBytes] = []
    for operand, moved in zip(operands, cost.traffic):
        by_level = bytes_by_storage(operand.type)
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


def _accumulate(
    flops: dict[str, int],
    flops_per_unit: dict[str, int],
    traffic: dict[str, TrafficBytes],
    record: ComputeCostMetadata,
) -> None:
    for name, value in record.flops:
        flops[name] = flops.get(name, 0) + value
    for name, value in record.flops_per_unit:
        flops_per_unit[name] = flops_per_unit.get(name, 0) + value
    for level, value in record.traffic:
        current = traffic.get(level, TrafficBytes())
        traffic[level] = TrafficBytes(
            current.read + value.read,
            current.write + value.write,
        )


def _scale_traffic(
    traffic: tuple[tuple[str, TrafficBytes], ...], trips: int
) -> tuple[tuple[str, TrafficBytes], ...]:
    if trips == 1:
        return traffic
    return tuple(
        (level, TrafficBytes(value.read * trips, value.write * trips))
        for level, value in traffic
    )


def _scale_moved(
    operands: tuple[TrafficBytes, ...], trips: int
) -> tuple[TrafficBytes, ...]:
    if trips == 1:
        return operands
    return tuple(
        TrafficBytes(item.read * trips, item.write * trips) for item in operands
    )


def _scaled(
    flops: tuple[tuple[str, int], ...],
    flops_per_unit: tuple[tuple[str, int], ...],
    traffic: tuple[tuple[str, TrafficBytes], ...],
    trips: int,
) -> ComputeCostMetadata:
    """A callee's totals, charged once per trip of the loops around the call."""
    return ComputeCostMetadata(
        flops=tuple((name, value * trips) for name, value in flops),
        flops_per_unit=tuple(
            (name, value * trips) for name, value in flops_per_unit
        ),
        traffic=_scale_traffic(traffic, trips),
    )


def _scaled_flops(flops: dict, trips: int) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted((dtype.name, value * trips) for dtype, value in flops.items())
    )


def _resolved_topologies(
    topologies: tuple[Topology, ...], target: Target
) -> tuple[Topology, ...]:
    """Fill launch-provided extents from the target's physical limits."""
    return tuple(
        topology
        if topology.size is not None
        else Topology(topology.name, target.topology_limit(topology.name))
        for topology in topologies
    )


def analyze_compute_cost(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach one work record per Call reachable from *function*.

    Callees are measured before their callers so a call site can report the
    callee's totals rather than re-walking its body.
    """
    topologies = _resolved_topologies(module.effective_topologies(), target)
    totals: dict[int, _Totals] = {}
    for fn in reversed(reachable_functions(function)):
        scope = FunctionScope(module, fn)
        whole = CostEvaluator(CostContext(scope=scope))
        local = CostEvaluator(
            CostContext(scope=scope, level=level, topologies=topologies)
        )
        flops: dict[str, int] = {}
        flops_per_unit: dict[str, int] = {}
        traffic: dict[str, TrafficBytes] = {}
        trips = enclosing_trips(fn.body)
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            count = trips.get(id(expr), 1)
            if isinstance(expr.target, Function):
                child = totals.get(id(expr.target))
                if child is None:
                    raise AnalysisError(
                        f"{describe(expr)}: recursive or unresolved Function call graph"
                    )
                record = _scaled(
                    child.flops, child.flops_per_unit, child.traffic, count
                )
            else:
                try:
                    whole_cost = whole.visit(expr)
                    local_cost = local.visit(expr)
                except (ValueError, VerifyError) as error:
                    raise AnalysisError(str(error)) from None
                traffic_by_level, operands = _call_movement(expr, whole_cost)
                record = ComputeCostMetadata(
                    flops=_scaled_flops(whole_cost.flops, count),
                    flops_per_unit=_scaled_flops(local_cost.flops, count),
                    traffic=_scale_traffic(traffic_by_level, count),
                    operands=_scale_moved(operands, count),
                )
            attach(expr, record)
            _accumulate(flops, flops_per_unit, traffic, record)
        totals[id(fn)] = _Totals(
            flops=tuple(sorted(flops.items())),
            flops_per_unit=tuple(sorted(flops_per_unit.items())),
            traffic=tuple(sorted(traffic.items())),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
