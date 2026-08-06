"""How much work the authored program asks for.

This family reads the program and nothing else. Flops come from each op's
registered cost evaluator and bytes from the logical types its operands and
result carry, so the record it leaves is the same on every backend. What that
work costs in time is a separate question, asked by the roofline family against
a target's rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import Target
from tilefoundry.visitor_registry import cost_evaluator_registry
from tilefoundry.visitor_registry.contexts import Cost, CostContext

from .errors import AnalysisError
from .metadata import ComputeCostMetadata, TrafficBytes
from .walk import (
    attach,
    bytes_by_storage,
    describe,
    enclosing_trips,
    execution_count,
    postorder,
    reachable_functions,
)

SELECTOR = "compute-cost"


@dataclass(frozen=True)
class _Totals:
    """One function's work, summed over the calls in its body."""

    flops: tuple[tuple[str, int], ...]
    traffic: tuple[tuple[str, TrafficBytes], ...]


def _local_cost(call: Call, module: Module) -> Cost:
    """The per-instance logical cost of *call*, from its cost evaluator."""
    evaluate = cost_evaluator_registry.lookup(type(call.target))
    if evaluate is None:
        raise AnalysisError(
            f"{describe(call)}: no cost evaluator registered for "
            f"{type(call.target).__name__}"
        )
    return evaluate(call, CostContext(module=module))


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
        # A Type spanning several levels states no split, so the aggregate
        # charges the whole Type at each, in the directions the op reported.
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
    traffic: dict[str, TrafficBytes],
    record: ComputeCostMetadata,
) -> None:
    for name, value in record.flops:
        flops[name] = flops.get(name, 0) + value
    for level, value in record.traffic:
        current = traffic.get(level, TrafficBytes())
        traffic[level] = TrafficBytes(
            current.read + value.read,
            current.write + value.write,
        )


def _call_record(
    call: Call,
    fn: Function,
    module: Module,
    topologies: tuple[Topology, ...],
    totals: dict[int, _Totals],
    trips: int = 1,
) -> ComputeCostMetadata:
    """The work record for one call site.

    A call into another Function costs what that Function costs. Its callee was
    measured first, so the totals are already there; their absence means the
    call graph is recursive or unresolved, which no amount of measuring fixes.

    *trips* is how many times the loops around this call run it. Unlike the mesh
    count it scales the traffic too, and for the opposite reason: a loop body's
    operand types are the tile one trip touches, so the whole tensor is the tile
    times the trips. A mesh's operand types are already the whole.
    """
    if isinstance(call.target, Function):
        child = totals.get(id(call.target))
        if child is None:
            raise AnalysisError(
                f"{describe(call)}: recursive or unresolved Function call graph"
            )
        return _scaled(child.flops, child.traffic, trips)
    count = execution_count(call, fn, topologies)
    cost = _local_cost(call, module)
    traffic, operands = _call_movement(call, cost)
    return ComputeCostMetadata(
        flops=tuple(
            sorted(
                (dtype.name, value * count * trips)
                for dtype, value in cost.flops.items()
            )
        ),
        traffic=_scale_traffic(traffic, trips),
        execution_count=count * trips,
        operands=_scale_moved(operands, trips),
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
    traffic: tuple[tuple[str, TrafficBytes], ...],
    trips: int,
) -> ComputeCostMetadata:
    """A callee's totals, charged once per trip of the loops around the call."""
    return ComputeCostMetadata(
        flops=tuple((name, value * trips) for name, value in flops),
        traffic=_scale_traffic(traffic, trips),
        execution_count=trips,
    )


def analyze_compute_cost(
    module: Module,
    function: Function,
    target: Target,
    options: object | None = None,
) -> None:
    """Attach one work record per Call reachable from *function*.

    Callees are measured before their callers so a call site can report the
    callee's totals rather than re-walking its body.
    """
    topologies = module.effective_topologies()
    totals: dict[int, _Totals] = {}
    for fn in reversed(reachable_functions(function)):
        flops: dict[str, int] = {}
        traffic: dict[str, TrafficBytes] = {}
        trips = enclosing_trips(fn.body)
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            record = _call_record(
                expr, fn, module, topologies, totals, trips.get(id(expr), 1)
            )
            attach(expr, record)
            _accumulate(flops, traffic, record)
        totals[id(fn)] = _Totals(
            flops=tuple(sorted(flops.items())),
            traffic=tuple(sorted(traffic.items())),
        )


__all__ = ["SELECTOR", "analyze_compute_cost"]
