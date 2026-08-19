"""What each occurrence moves, and whose bytes it moves them into.

The movement half of an Op's registered cost, and the alias proof that decides
whether a value was given bytes of its own. Both belong to the family that
knows where values live: how much crossed a boundary is the Op's own answer,
and whether that crossing was a copy is a question about allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Call, Constant, Expr, VerifyError
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    StorageEffectClaim,
    StorageEffectKind,
    access_elements,
    access_relation_registry,
    declared_storage,
    elements_of,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .metadata import BufferAliasMetadata, TrafficBytes, TrafficMetadata
from .walk import bytes_by_storage, children, describe, postorder, tensor_types

_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


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


def _stated_movement(call: Call, cost: Cost, ctx: CostContext) -> tuple[TrafficBytes, ...] | None:
    """Per-operand movement as the Op's own access relation states it.

    A Type says how big a value is, not how much of it this occurrence touches.
    The two agree while every operand is sharded the way the result is, and part
    company the moment one is not. The relation is asked instead, in this
    context's window, so one handler answers for the whole program and for one
    unit. Only the amount comes from it: which direction an operand moves stays
    the cost's answer. ``None`` for an Op with no relation yet.
    """
    handler = access_relation_registry.lookup(type(call.target))
    if handler is None:
        return None
    relations = handler(call, ctx)
    operands = (*call.args, call)
    stated: list[TrafficBytes] = []
    for index, (operand, moved) in enumerate(zip(operands, cost.traffic)):
        if index == len(call.args):
            moving = _output_bytes(relations, ctx.local_type_of(call))
        else:
            quantity = access_elements(relations, boundary=index)
            moving = _bytes_for(
                ctx.local_type_of(operand),
                quantity.upper if quantity is not None else None,
            )
        if moving is None:
            stated.append(moved)
            continue
        stated.append(
            TrafficBytes(moving if moved.read else 0, moving if moved.write else 0)
        )
    return tuple(stated)


def _output_bytes(relations: AccessRelations, held: Type) -> int | None:
    """Bytes the result moves, taking one output boundary per field it has.

    A tuple result is as many boundaries as it has fields, each somewhere of its
    own that the Op stated separately. Reading only the first would drop the
    rest, and reading the tuple as one value has no element count to read at
    all -- either way the Op's own answer is thrown away for a Type's.
    """
    fields = held.fields if isinstance(held, TupleType) else (held,)
    total = 0
    for position, field_ in enumerate(fields):
        quantity = access_elements(relations, boundary=position, output=True)
        moving = _bytes_for(field_, quantity.upper if quantity is not None else None)
        if moving is None:
            return None
        total += moving
    return total


def _bytes_for(held: Type, elements: int | None) -> int | None:
    """The bytes *elements* of *held* occupy, or ``None`` when unanswerable.

    Taken as a share of the whole rather than as an element size, because a
    packed dtype has no whole number of bytes per element and a bool boundary
    would round to nothing.
    """
    if elements is None or not isinstance(held, TensorType):
        return None
    try:
        whole = elements_of(held)
    except ValueError:
        return None
    if whole <= 0:
        return 0
    return tensor_bytes(held) * elements // whole


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

    def local_type_of(self, expr: Expr) -> Type:
        """Types as written: a storage claim is about the whole value.

        Where a value's bytes live is one fact for the program, not one per
        participant, so this walk asks the relations in no topology window.
        """
        return self.type_of(expr)

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
    return Cost(cost.flops, tuple(traffic), cost.service)


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
    return Cost(cost.flops, tuple(traffic), cost.service)


def call_traffic(
    expr: Call,
    whole: CostEvaluator,
    local: CostEvaluator,
    settled: "StorageConclusion | None" = None,
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. What the Op states is corrected by what the
    alias proof settled: bytes a forwarding operation was shown not to move are
    not charged, and the operands keep their own share of what remains.
    """
    try:
        whole_cost = whole.visit(expr)
        local_cost = local.visit(expr)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    local_types = (
        *(local.ctx.local_type_of(arg) for arg in expr.args),
        local.ctx.local_output_type(expr),
    )
    stated = _stated_movement(expr, local_cost, local.ctx)
    if stated is not None:
        local_cost = Cost(local_cost.flops, stated, local_cost.service)
    if settled is not None:
        whole_cost = _aliased_cost(expr, whole_cost, settled, whole.ctx.type_of(expr))
        local_cost = _aliased_cost(expr, local_cost, settled, local_types[-1])
    traffic_by_level, operands = _call_movement(expr, whole_cost)
    local_traffic, _local_operands = _call_movement(expr, local_cost, local_types)
    return TrafficMetadata(
        whole=traffic_by_level,
        per_unit=local_traffic,
        operands=operands,
    )


def add_traffic(
    whole: dict[str, TrafficBytes],
    per_unit: dict[str, TrafficBytes],
    record: TrafficMetadata,
    trips: int,
) -> None:
    """Add one occurrence's bytes to a function's, as often as it happens."""
    for into, stated in ((whole, record.whole), (per_unit, record.per_unit)):
        for level, moved in stated:
            running = into.get(level, TrafficBytes())
            into[level] = TrafficBytes(
                running.read + moved.read * trips,
                running.write + moved.write * trips,
            )
