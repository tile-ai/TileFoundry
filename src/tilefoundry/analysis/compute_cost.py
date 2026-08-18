"""How much work the authored program asks for.

This family reads the program and nothing else. Flops come from each op's
registered cost evaluator and bytes from the logical types its operands and
result carry, so the record it leaves is the same on every backend. What that
work costs in time is a separate question, asked by the roofline family against
a target's rates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tilefoundry.ir.core import Call, Constant, Expr, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, Type, tensor_bytes
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import Target
from tilefoundry.visitor_registry.access_relation import (
    StorageEffectClaim,
    StorageEffectKind,
    declared_storage,
    static_bytes,
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
            f"performance: selected topology level {level!r}, but the target's "
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
            raise AnalysisError(f"performance: unknown compute dtype {name!r}")
        rate = facts.peak_per_unit_for(dtype)
        if rate is None or rate <= 0:
            raise AnalysisError(
                f"performance: target publishes no per-unit compute rate for "
                f"dtype {name!r} at {level!r}"
            )
        compute_ns += -(-(value * scale * 1_000_000_000) // rate)

    moved = cost.traffic_per_unit_at(facts.bandwidth_level).total_bytes * scale
    memory_ns = 0
    if moved:
        rate = facts.memory_bandwidth_bytes_per_second_per_unit
        if rate is None or rate <= 0:
            raise AnalysisError(
                f"performance: target publishes no per-unit bandwidth for level "
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


@dataclass(frozen=True)
class _Extent:
    """Where one value's bytes sit, as an offset and length in a base buffer."""

    base: Expr
    offset: int
    size: int


@dataclass
class _Storage:
    """What one walk of a Function has settled about where values live."""

    type_of: Callable[[Expr], Type]
    users: dict[int, list[Expr]]
    positions: dict[int, int]
    caller_owned: frozenset[int]
    bases: dict[int, Expr] = field(default_factory=dict)
    extents: dict[int, _Extent] = field(default_factory=dict)
    members: dict[int, list[Expr]] = field(default_factory=dict)
    claims: dict[int, StorageEffectClaim | None] = field(default_factory=dict)

    def claim_of(self, call: Call) -> StorageEffectClaim | None:
        """What the Op says about this Call, asked once.

        Asking builds the Op's relations, and three readers want the same
        answer: what it claims, what it was aiming at, and whether it may.
        """
        if id(call) not in self.claims:
            self.claims[id(call)] = declared_storage(call, self)
        return self.claims[id(call)]

    def record(self, expr: Expr, base: Expr, extent: _Extent | None = None) -> None:
        """Remember that *expr*'s bytes belong to *base*, at *extent* if known.

        Keyed by identity, like everything else here: two values that look alike
        are still two values, and an `Expr` compares and hashes by structure.

        Which base a value belongs to and where inside it are separate answers:
        a claim can settle the first without the second, and everything that
        asks who else shares this buffer needs the first either way.
        """
        self.bases[id(expr)] = base
        if extent is not None:
            self.extents[id(expr)] = extent
        self.members.setdefault(id(base), []).append(expr)

    def extent_of(self, expr: Expr) -> _Extent | None:
        """Where *expr* sits in a base, or ``None`` when no address is known."""
        known = self.extents.get(id(expr))
        if known is not None:
            return known
        if id(expr) in self.bases:
            return None
        size = static_bytes(self.type_of(expr))
        return None if size is None else _Extent(expr, 0, size)

    def base_of(self, expr: Expr) -> Expr:
        """The value that owns the bytes *expr* reads."""
        return self.bases.get(id(expr), expr)

    def last_authored_use(self, base: Expr, call: Call) -> bool:
        """Whether *call* is the last authored reader of *base*'s alias group."""
        here = self.positions.get(id(call))
        if here is None:
            return False
        for member in (base, *self.members.get(id(base), ())):
            for user in self.users.get(id(member), ()):
                if user is not call and self.positions.get(id(user), -1) > here:
                    return False
        return True


@dataclass(frozen=True)
class StorageConclusion:
    """What one Call's storage came to, and what it had been aiming at.

    A write whose proof did not close still overwrote nothing, so it owes the
    part of the container it did not touch. That is the same claim the proof
    read, so it travels with the conclusion rather than being asked for again.
    """

    alias: BufferAliasMetadata
    destination: int | None = None


def alias_conclusions(fn: Function, evaluator: CostEvaluator) -> dict[int, StorageConclusion]:
    """Decide, in authored order, where every Call's result bytes live.

    The whole function is indexed first because an in-place write is only sound
    once nothing that shares the destination will read it again, and that is not
    a fact about the write itself.
    """
    values = postorder(fn.body)
    users: dict[int, list[Expr]] = {}
    for expr in values:
        for child in children(expr):
            users.setdefault(id(child), []).append(expr)
    walk = _Storage(
        type_of=evaluator.ctx.type_of,
        users=users,
        positions={id(expr): index for index, expr in enumerate(values)},
        caller_owned=frozenset(id(param) for param in fn.params),
    )
    conclusions: dict[int, StorageConclusion] = {}
    for expr in values:
        if not isinstance(expr, Call):
            continue
        proven = _prove_storage(expr, walk)
        claim = walk.claim_of(expr)
        conclusions[id(expr)] = StorageConclusion(
            alias=(
                BufferAliasMetadata()
                if proven is None
                else BufferAliasMetadata(proven[0].value, proven[1])
            ),
            destination=(
                claim.operands[0]
                if claim is not None
                and claim.kind is StorageEffectKind.UPDATE
                and len(claim.operands) == 1
                else None
            ),
        )
    return conclusions


def _prove_storage(
    call: Call, walk: _Storage
) -> tuple[StorageEffectKind, tuple[int, ...]] | None:
    """Ask the Op what it claims, and hold that claim to one base.

    Spans may reach fewer operands than the claim names, so a resolved coverage
    has to land on that same base: a conclusion covers every operand it names,
    and its reader retires the movement of each. An overwrite is admitted only
    once nothing that shares those bytes will read them again and they are this
    function's to reuse, which is a fact about the walk and not about the Op.
    """
    claim = walk.claim_of(call)
    if claim is None or not claim.operands:
        return None
    base = _one_base(call, claim.operands, walk)
    if base is None:
        return None
    if claim.kind is StorageEffectKind.UPDATE and not _may_overwrite(call, base, walk):
        return None
    extent = _compose(call, claim, walk) if claim.spans else None
    if extent is not None:
        if extent.base is not base:
            return None
        walk.record(call, base, extent)
        return claim.kind, claim.operands
    if claim.spans_required:
        return None
    walk.record(call, base)
    return claim.kind, claim.operands


def _may_overwrite(call: Call, base: Expr, walk: _Storage) -> bool:
    """Whether this function may reuse *base*'s bytes at this call.

    A caller's parameter is not this function's to reuse, and neither is a
    constant. Every other operand has to be somewhere else, or the write would
    be reading what it is about to replace.
    """
    if id(base) in walk.caller_owned or isinstance(base, Constant):
        return False
    destinations = _destinations(call, walk)
    for index, operand in enumerate(call.args):
        if index not in destinations and walk.base_of(operand) is base:
            return False
    return walk.last_authored_use(base, call)


def _destinations(call: Call, walk: _Storage) -> frozenset[int]:
    """Which operand positions an update claims as its destination."""
    claim = walk.claim_of(call)
    if claim is None or claim.kind is not StorageEffectKind.UPDATE:
        return frozenset()
    return frozenset(claim.operands)


def _one_base(call: Call, operands: tuple[int, ...], walk: _Storage) -> Expr | None:
    """The single base every named operand resolves to, if there is one."""
    if not all(0 <= operand < len(call.args) for operand in operands):
        return None
    bases = [walk.base_of(call.args[operand]) for operand in operands]
    first = bases[0]
    return first if all(base is first for base in bases) else None


def _compose(call: Call, claim: StorageEffectClaim, walk: _Storage) -> _Extent | None:
    """Resolve a claim's spans onto one base, covering the result exactly.

    The spans are read in the order the result is laid out in, and each has to
    begin where the previous one ended. Sorting them first would accept a claim
    whose pieces are all there but in the wrong places, which is a different
    result than the one being proven.
    """
    result_bytes = static_bytes(walk.type_of(call))
    if result_bytes is None:
        return None
    resolved: list[tuple[Expr, int, int]] = []
    for span in claim.spans:
        if not 0 <= span.operand < len(call.args) or span.offset < 0 or span.size <= 0:
            return None
        operand = walk.extent_of(call.args[span.operand])
        if operand is None or span.offset + span.size > operand.size:
            return None
        resolved.append((operand.base, operand.offset + span.offset, span.size))
    base = resolved[0][0]
    if any(item[0] is not base for item in resolved):
        return None
    start = cursor = resolved[0][1]
    for _base, offset, size in resolved:
        if offset != cursor:
            return None
        cursor += size
    if cursor - start != result_bytes:
        return None
    return _Extent(base, start, result_bytes)


def _aliased_cost(
    call: Call, cost: Cost, settled: StorageConclusion, result_type: Type
) -> Cost:
    """Correct one operation's own cost by what the alias proof concluded.

    An operation that reports the copy it would make -- a transpose, a
    concatenation -- retires those bytes once its result is shown to be where
    they already were. An in-place write that failed its proof gains the other
    direction: it has to carry the part of the container it did not touch into a
    result of its own. Every other operation already reports what it moves.
    """
    alias = settled.alias
    if alias.kind == "forward" and alias.aliased_operands:
        return _without_forwarded_movement(call, cost, alias.aliased_operands)
    if alias.kind == "produce" and settled.destination is not None:
        return _with_untouched_copy(call, cost, settled.destination, result_type)
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
    settled: StorageConclusion | None = None,
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
    if settled is not None:
        whole_cost = _aliased_cost(expr, whole_cost, settled, whole.ctx.type_of(expr))
        local_cost = _aliased_cost(expr, local_cost, settled, local_types[-1])
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
            settled = aliases[id(expr)]
            attach(expr, settled.alias)
            record = _call_cost_record(expr, whole, local, settled)
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
