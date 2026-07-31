"""How long the requested work must take, at best.

This family adds no measurement of its own. It reads the work the compute-cost
records state and divides it by the rates the target publishes, so a change here
can only ever be a change in how the two are combined -- never a second, subtly
different count of the same flops.
"""

from __future__ import annotations

import math

from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.cuda.target import CudaTarget
from tilefoundry.target.facts import TARGET_FACTS

from .errors import AnalysisError
from .facts import ThroughputFacts
from .metadata import ComputeCostMetadata, RooflineMetadata, TrafficBytes
from .registry import register_analysis
from .walk import attach, describe, postorder, reachable_functions

SELECTOR = "roofline"

# No published rate means no bound from that side, which is reported as zero
# rather than as a guess.
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
        total += math.ceil(value * 1_000_000_000 / rate)
    return total


def _memory_ns(traffic: TrafficBytes, facts: ThroughputFacts) -> int:
    """Time the traffic at the target's bandwidth level needs."""
    if facts.memory_bandwidth_bytes_per_second is None or not traffic.total_bytes:
        return _NO_BOUND
    return math.ceil(
        traffic.total_bytes
        * 1_000_000_000
        / facts.memory_bandwidth_bytes_per_second
    )


def _bound(compute_ns: int, memory_ns: int, *, has_work: bool) -> RooflineMetadata:
    """Combine the two sides into one bound and name the one that set it.

    Work the machine has no published rate for still takes time, so a call that
    does something reports at least one nanosecond. Reporting zero would read as
    free.
    """
    theoretical = max(compute_ns, memory_ns)
    if not theoretical:
        theoretical = 1 if has_work else 0
    if not theoretical:
        bound_by = "none"
    elif compute_ns and compute_ns == memory_ns:
        # Calling an exact tie one side's win would hide that relieving that
        # side alone buys nothing.
        bound_by = "balanced"
    elif memory_ns > compute_ns:
        bound_by = "memory"
    elif compute_ns:
        bound_by = "compute"
    else:
        # Work whose rate the target does not publish still takes time, and
        # saying which side that time came from would be an invention.
        bound_by = "unrated"
    return RooflineMetadata(
        compute_ns=compute_ns,
        memory_ns=memory_ns,
        theoretical_ns=theoretical,
        bound_by=bound_by,
    )


def analyze_roofline(
    module: Module,
    function: Function,
    target: object,
    options: object | None = None,
) -> None:
    """Attach a bound to every Call, and one to every Function, reachable here."""
    facts = TARGET_FACTS.project(target, ThroughputFacts)
    for fn in reachable_functions(function):
        total_flops: dict[str, int] = {}
        total_traffic = TrafficBytes()
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            cost = get_metadata(expr, ComputeCostMetadata)
            if cost is None:
                raise AnalysisError(
                    f"{describe(expr)}: roofline needs the compute-cost record "
                    "this call was never given"
                )
            traffic = cost.traffic_at(facts.bandwidth_level)
            attach(
                expr,
                _bound(
                    _compute_ns(cost.flops, facts),
                    _memory_ns(traffic, facts),
                    has_work=bool(cost.flops or cost.traffic),
                ),
            )
            for name, value in cost.flops:
                total_flops[name] = total_flops.get(name, 0) + value
            total_traffic = TrafficBytes(
                total_traffic.read + traffic.read,
                total_traffic.write + traffic.write,
            )
        # The function's bound sums each side first and compares once. Taking the
        # per-Call bounds and adding them would charge the machine twice for work
        # its two halves do at the same time.
        attach(
            fn,
            _bound(
                _compute_ns(tuple(sorted(total_flops.items())), facts),
                _memory_ns(total_traffic, facts),
                has_work=bool(total_flops or total_traffic.total_bytes),
            ),
        )


for _target_type in (CudaTarget, AmxTarget):
    register_analysis(
        _target_type,
        SELECTOR,
        requires=("memory", "compute-cost"),
        produces=(RooflineMetadata,),
    )(analyze_roofline)


__all__ = ["SELECTOR", "analyze_roofline"]
