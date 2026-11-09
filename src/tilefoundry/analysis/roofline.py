"""How long the requested work must take, at best.

This family adds no measurement of its own. It reads the work the compute-cost
records state and divides it by the rates the target publishes, so a change here
can only ever be a change in how the two are combined -- never a second, subtly
different count of the same flops.
"""

from __future__ import annotations

from tilefoundry import target
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr

from .compute_cost import inflight_bytes
from .errors import Error
from .metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
)
from .visitor import AnalyzeContext

SELECTOR = "roofline"


NO_BOUND = 0


def totals_of(held) -> tuple[tuple[str, int], ...]:
    """One typed work category's whole-program counts."""
    return tuple((name, spread.total) for name, spread in held.kinds)


def compute_time_ns(flops: tuple[tuple[str, int], ...], facts: target.ThroughputFacts) -> int:
    """Time the flops need, summed over the dtypes that have a rate.

    A dtype the target publishes no rate for contributes nothing. That
    under-reports the bound, which is the safe direction for a lower bound:
    claiming time the hardware never promised would be worse.
    """
    total = 0
    for name, value in flops:
        if not value:
            continue
        dtype = getattr(expr.DType, name, None)
        if dtype is None:
            raise Error(f"roofline: unknown compute dtype {name!r}")
        rate = facts.peak_for(dtype)
        if rate is None:
            continue
        total += -(-(value * 1_000_000_000) // rate)
    return total


def memory_time_ns(crossed: int, facts: target.ThroughputFacts) -> int:
    """Time the bytes at the target's bandwidth level need."""
    if facts.memory_bandwidth_bytes_per_second is None or not crossed:
        return NO_BOUND
    return -(-(crossed * 1_000_000_000) // facts.memory_bandwidth_bytes_per_second)


def bound_of(compute_ns: int, memory_ns: int, *, has_work: bool) -> RooflineMetadata:
    """Combine the two sides into one bound and name the one that set it.

    A nanosecond is owed by what this could have priced: work of a rated kind
    whose own rate is missing reports at least one, because reporting zero for
    it would read as free. Bytes at a level with no published bandwidth owe
    nothing -- no rate was ever stated for them, so no floor follows.
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


def cost_bound(
    cost: ComputeCostMetadata,
    moved: MemoryMetadata | RegionMemoryMetadata,
    facts: target.ThroughputFacts,
) -> RooflineMetadata:
    """Bound one occurrence from the work it does and the bytes it moves.

    Whole-program work against the rates of the units the program runs: the
    flops the target publishes a peak for, and the footprint's bytes at the
    level it publishes a bandwidth for, reads and writes as one count. Typed
    service has no whole-device rate to divide by and so does not enter a bound,
    and neither do bytes at a level with no published bandwidth: what asks for a
    nanosecond is what this could have priced, so a dtype whose rate is missing
    still owes one and a level nobody rated does not.
    """
    reached = moved.traffic.storage.of(facts.bandwidth_level)
    crossed = 0 if reached is None else reached.total.total_bytes
    unique = None if moved.footprint is None else moved.footprint.bytes_at(facts.bandwidth_level)
    crossed = crossed if unique is None else unique[1]
    return bound_of(
        compute_time_ns(totals_of(cost.flops), facts),
        memory_time_ns(crossed, facts),
        has_work=bool(any(value for _, value in totals_of(cost.flops)) or crossed),
    )


class RooflineVisitor(valuation.Visitor[None]):
    """Attach one bound per Call in expression order.

    Each Call is priced for the bytes it keeps in flight; the Function, for the
    most any of its own Calls keeps, which keeps its bound a lower bound. A call
    into another definition is not an access: that definition prices its own.
    """

    def __init__(self, query) -> None:
        super().__init__()
        self.query, self.inflight = query, 0

    def facts(self, ctx: AnalyzeContext, inflight: int | None) -> target.ThroughputFacts:
        query = None if self.query is None else (*self.query, inflight)
        return ctx.target.get_facts(target.ThroughputFacts, query)

    def default_visit_leaf(
        self, node: expr.Expr, operands: tuple[None, ...], ctx: AnalyzeContext
    ) -> None:
        if not isinstance(node, expr.Call):
            return
        cost = expr.get_metadata(node, ComputeCostMetadata)
        if cost is None:
            raise Error(
                f"{expr.describe_value(node)}: roofline needs the compute-cost record "
                "this call was never given"
            )
        moved = expr.get_metadata(node, MemoryMetadata)
        if moved is None:
            raise Error(
                f"{expr.describe_value(node)}: roofline needs the traffic record the "
                "memory family states for every call it measures"
            )
        level = ctx.target.get_facts(target.ThroughputFacts).bandwidth_level
        callee = ctx.callee_context(node) is not None
        inflight = None if callee else inflight_bytes(node, ctx.scope, ctx.target, level)
        self.inflight = max(self.inflight, inflight or 0)
        node.attach_metadata(cost_bound(cost, moved, self.facts(ctx, inflight)))


def analyze_roofline(
    function: core.Function,
    context: AnalyzeContext,
) -> None:
    """Attach a bound to every Call and the normalized Function."""
    unit = context.topology_level
    visitor = RooflineVisitor(None if unit is None else (unit, context.scope.active_positions(unit)))
    visitor.visit(function.body, context)
    total = expr.get_metadata(function, ComputeCostMetadata)
    if total is None:
        raise Error(
            f"function {function.name!r}: roofline needs the compute-cost root "
            "record this function was never given"
        )
    moved = expr.get_metadata(function, RegionMemoryMetadata)
    if moved is None:
        raise Error(
            f"function {function.name!r}: roofline needs the traffic root record "
            "the memory family states for every function it measures"
        )
    function.attach_metadata(cost_bound(total, moved, visitor.facts(context, visitor.inflight)))


__all__ = ["SELECTOR", "analyze_roofline"]
