"""How long the requested work must take, at best.

This family adds no measurement of its own. It reads the work the compute-cost
records state and divides it by the rates the target publishes, so a change here
can only ever be a change in how the two are combined -- never a second, subtly
different count of the same flops.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType
from tilefoundry.target import Target

from .errors import AnalysisError
from .facts import ThroughputFacts
from .metadata import ComputeCostMetadata, RooflineMetadata, TrafficBytes
from .walk import attach, describe, postorder, reachable_functions

SELECTOR = "roofline"



_NO_BOUND = 0


def _compute_ns(
    flops: tuple[tuple[str, int], ...], facts: ThroughputFacts
) -> int:
    """Time the flops need, summed over the dtypes that have a rate.

    A dtype the target publishes no rate for contributes nothing. That
    under-reports the bound, which is the safe direction for a lower bound:
    claiming time the hardware never promised would be worse.
    """
    total = 0
    for name, value in flops:
        if not value:
            continue
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise AnalysisError(f"roofline: unknown compute dtype {name!r}")
        rate = facts.peak_for(dtype)
        if rate is None:
            continue
        total += -(-(value * 1_000_000_000) // rate)
    return total


def _memory_ns(traffic: TrafficBytes, facts: ThroughputFacts) -> int:
    """Time the traffic at the target's bandwidth level needs."""
    if facts.memory_bandwidth_bytes_per_second is None or not traffic.total_bytes:
        return _NO_BOUND
    numerator = traffic.total_bytes * 1_000_000_000
    return -(-numerator // facts.memory_bandwidth_bytes_per_second)


def _bound(compute_ns: int, memory_ns: int, *, has_work: bool) -> RooflineMetadata:
    """Combine the two sides into one bound and name the one that set it.

    Work the machine has no published rate for still takes time, so a call that
    does something reports at least one nanosecond. Reporting zero would read as
    free.
    """
    ideal = max(compute_ns, memory_ns)
    if not ideal:
        ideal = 1 if has_work else 0
    if not ideal:
        bound_by = "none"
    elif compute_ns and compute_ns == memory_ns:


        bound_by = "balanced"
    elif memory_ns > compute_ns:
        bound_by = "memory"
    elif compute_ns:
        bound_by = "compute"
    else:


        bound_by = "unrated"
    return RooflineMetadata(
        compute_ns=compute_ns,
        memory_ns=memory_ns,
        ideal_ns=ideal,
        bound_by=bound_by,
    )


def _cost_bound(
    cost: ComputeCostMetadata, facts: ThroughputFacts, *, scale: int = 1
) -> RooflineMetadata:
    """Bound one compute-cost record after applying an execution count."""
    flops = tuple((name, value * scale) for name, value in cost.flops)
    traffic = cost.traffic_at(facts.bandwidth_level)
    traffic = TrafficBytes(traffic.read * scale, traffic.write * scale)
    return _bound(
        _compute_ns(flops, facts),
        _memory_ns(traffic, facts),
        has_work=bool(cost.flops or cost.traffic),
    )


def analyze_roofline(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach a bound to every Call, and one to every Function, reachable here."""
    facts = target.get_facts(ThroughputFacts)
    for fn in reachable_functions(function):
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            cost = get_metadata(expr, ComputeCostMetadata)
            if cost is None:
                raise AnalysisError(
                    f"{describe(expr)}: roofline needs the compute-cost record "
                    "this call was never given"
                )
            attach(expr, _cost_bound(cost, facts))
        total = get_metadata(fn, ComputeCostMetadata)
        if total is None:
            raise AnalysisError(
                f"function {fn.name!r}: roofline needs the compute-cost root "
                "record this function was never given"
            )
        attach(fn, _cost_bound(total, facts))


__all__ = ["SELECTOR", "analyze_roofline"]
