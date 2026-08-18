"""How much work the authored program asks for.

This family reads the program and nothing else. Flops come from each op's
registered cost evaluator and bytes from the logical types its operands and
result carry, so the record it leaves is the same on every backend. What that
work costs in time is a separate question, asked by the roofline family against
a target's rates.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Expr, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, Type, tensor_bytes
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import Target
from tilefoundry.visitor_registry.alias import (
    AliasContext,
    declared_alias,
    prove_alias,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .facts import ThroughputFacts
from .metadata import BufferAliasMetadata, ComputeCostMetadata, TrafficBytes
from .walk import (
    attach,
    bytes_by_storage,
    children,
    describe,
    enclosing_trips,
    postorder,
    reachable_functions,
    tensor_types,
)

SELECTOR = "compute-cost"
_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


def _is_structural_occurrence(
    cost: ComputeCostMetadata,
    facts: ThroughputFacts,
) -> bool:
    """Whether an occurrence asks for nothing this model puts on a clock.

    Only the quantities that carry time are read: every dtype's work, and the
    movement at the one level the target states a bandwidth for. Movement at
    another level is recorded but is nobody's service here, so an occurrence
    that has only that still takes no modeled time.
    """
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

    moved = cost.traffic_per_unit_at(facts.bandwidth_level).total_bytes * scale
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
    Concrete leaves use their declared levels; a UMAT leaf gets the level at
    which this call consumes it. The result is deliberately not a consuming
    argument, even when it is UMAT.
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
    for index, (type_, moved) in enumerate(zip(types, cost.traffic)):
        is_call_arg = index < len(call.args)
        has_umat = any(
            tensor.storage is StorageKind.UMAT for tensor in tensor_types(type_)
        )
        by_level = bytes_by_storage(
            type_,
            umat_level=_UMAT_CONSUMPTION_LEVEL if is_call_arg else None,
        )
        charged.append(moved)
        if len(by_level) == 1 and not has_umat:
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


def alias_conclusions(fn: Function, evaluator: CostEvaluator) -> dict[int, BufferAliasMetadata]:
    """Decide, in authored order, where every Call's result bytes live.

    The whole function is indexed first because an in-place write is only sound
    once nothing that shares the destination will read it again, and that is not
    a fact about the write itself.
    """
    values = postorder(fn.body)
    positions = {id(expr): index for index, expr in enumerate(values)}
    users: dict[int, list[Expr]] = {}
    for expr in values:
        for child in children(expr):
            users.setdefault(id(child), []).append(expr)
    ctx = AliasContext(
        type_of=evaluator.ctx.type_of,
        users={key: tuple(value) for key, value in users.items()},
        positions=positions,
        caller_owned=frozenset(id(param) for param in fn.params),
    )
    conclusions: dict[int, BufferAliasMetadata] = {}
    for expr in values:
        if not isinstance(expr, Call):
            continue
        proven = prove_alias(expr, ctx)
        conclusions[id(expr)] = (
            BufferAliasMetadata()
            if proven is None
            else BufferAliasMetadata(proven[0].value, proven[1])
        )
    return conclusions


def _aliased_cost(
    call: Call, cost: Cost, alias: BufferAliasMetadata, result_type: Type
) -> Cost:
    """Correct one operation's own cost by what the alias proof concluded.

    An operation that reports the copy it would make -- a transpose, a
    concatenation -- retires those bytes once its result is shown to be where
    they already were. An in-place write that failed its proof gains the other
    direction: it has to carry the part of the container it did not touch into a
    result of its own. Every other operation already reports what it moves.
    """
    if alias.kind == "forward" and alias.aliased_operands:
        return _without_forwarded_movement(call, cost, alias.aliased_operands)
    destination = declared_alias(call.target)
    if alias.kind == "produce" and destination is not None and destination.destination is not None:
        return _with_untouched_copy(call, cost, destination.destination, result_type)
    return cost


def _without_forwarded_movement(
    call: Call, cost: Cost, operands: tuple[int, ...]
) -> Cost:
    """Retire the read of each forwarded operand and the write it fed."""
    traffic = list(cost.traffic)
    retired = 0
    for index in operands:
        retired += traffic[index].read
        traffic[index] = TrafficBytes(0, traffic[index].write)
    result = traffic[-1]
    if retired > result.write:
        raise AnalysisError(
            f"{describe(call)}: the alias proof retires {retired} B of a "
            f"{result.write} B result"
        )
    traffic[-1] = TrafficBytes(result.read, result.write - retired)
    return Cost(cost.flops, tuple(traffic))


def _with_untouched_copy(
    call: Call, cost: Cost, destination: int, result_type: Type
) -> Cost:
    """Charge the part of the container a materialized update has to carry."""
    traffic = list(cost.traffic)
    whole = tensor_bytes(result_type)
    untouched = whole - traffic[-1].write
    if untouched < 0:
        raise AnalysisError(
            f"{describe(call)}: the update writes {traffic[-1].write} B of a "
            f"{whole} B result"
        )
    moved = traffic[destination]
    traffic[destination] = TrafficBytes(moved.read + untouched, moved.write)
    traffic[-1] = TrafficBytes(traffic[-1].read, whole)
    return Cost(cost.flops, tuple(traffic))


def _call_cost_record(
    expr: Call,
    whole: CostEvaluator,
    local: CostEvaluator,
    alias: BufferAliasMetadata | None = None,
) -> ComputeCostMetadata:
    """Measure one Call without attaching the resulting record."""
    try:
        whole_cost = whole.visit(expr)
        local_cost = local.visit(expr)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    local_types = (
        *(local.ctx.local_type_of(arg) for arg in expr.args),
        local.ctx.local_output_type(expr),
    )
    if alias is not None:
        whole_cost = _aliased_cost(expr, whole_cost, alias, whole.ctx.type_of(expr))
        local_cost = _aliased_cost(expr, local_cost, alias, local_types[-1])
    traffic_by_level, operands = _call_movement(expr, whole_cost)
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
        aliases = alias_conclusions(fn, whole)
        for expr in postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            count = trips.get(id(expr), 1)
            alias = aliases[id(expr)]
            attach(expr, alias)
            record = _call_cost_record(expr, whole, local, alias)
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
