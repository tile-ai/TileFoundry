"""How long the requested work must take, at best.

This family adds no measurement of its own. It reads the work the compute-cost
records state and the bytes the memory family's traffic records state, and
divides each by the rates the target publishes, so a change here can only ever
be a change in how the two are combined -- never a second, subtly different
count of the same flops or the same bytes.
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
from .traffic import TrafficMetadata
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


def _moved_at(moved: TrafficMetadata, level: str) -> TrafficBytes:
    """The bytes a traffic record states at *level*, none being none moved."""
    return next(
        (value for name, value in moved.whole if name == level), TrafficBytes()
    )


def _cost_bound(
    cost: ComputeCostMetadata, moved: TrafficMetadata, facts: ThroughputFacts
) -> RooflineMetadata:
    """Bound one occurrence from the work it does and the bytes it moves."""
    return _bound(
        _compute_ns(cost.flops, facts),
        _memory_ns(_moved_at(moved, facts.bandwidth_level), facts),
        has_work=bool(cost.flops or moved.whole),
    )


def _traffic_of(fn: Function) -> TrafficMetadata:
    """One function's own total, refused when it was never settled."""
    moved = get_metadata(fn, TrafficMetadata)
    if moved is None:
        raise AnalysisError(
            f"function {fn.name!r}: roofline needs the traffic record the memory "
            "family states only when every occurrence in it gave an answer"
        )
    return moved


def _moved_by(expr: Call) -> TrafficMetadata:
    """The bytes one occurrence moves, a call standing for what it calls."""
    if isinstance(expr.target, Function):
        return _traffic_of(expr.target)
    moved = get_metadata(expr, TrafficMetadata)
    if moved is None:
        raise AnalysisError(
            f"{describe(expr)}: roofline needs the traffic record this call was "
            "never given, and its flops alone do not bound it"
        )
    return moved


def analyze_roofline(
    module: Module,
    function: Function,
    target: Target,
    level: str | None = None,
    options: object | None = None,
) -> None:
    """Attach a bound to every Call, and one to every Function, reachable here.

    A bound is a claim about how fast something can go, so it is made only where
    both halves of it were stated. An occurrence whose bytes nobody could state
    is refused rather than bounded by its flops alone: reporting the compute side
    of a missing memory side reads as a call that turned out to be compute-bound.
    """
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
            attach(expr, _cost_bound(cost, _moved_by(expr), facts))
        total = get_metadata(fn, ComputeCostMetadata)
        if total is None:
            raise AnalysisError(
                f"function {fn.name!r}: roofline needs the compute-cost root "
                "record this function was never given"
            )
        attach(fn, _cost_bound(total, _traffic_of(fn), facts))


__all__ = ["SELECTOR", "analyze_roofline"]
