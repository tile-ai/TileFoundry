"""What each occurrence moves.

The movement half of an Op's registered cost: how much crossed each boundary,
which is what that boundary's own relation reaches, and at which storage level
those bytes are charged, which is where the leaves it reached live.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    leaves_of,
    reached_elements,
    reached_leaves,
    relations_of,
    static_bytes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .metadata import TrafficBytes, TrafficMetadata
from .walk import bytes_by_storage, describe

_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


def _call_movement(
    call: Call,
    cost: Cost,
    charged: tuple[dict[str, int], ...],
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and where those bytes are charged.

    How much moves is what that boundary's own relation reaches; which level it
    moves at is where the leaf it reached lives. *charged* is that reading
    already split by level, one mapping per operand, so a Type is never
    re-expanded here: an operand reaching one leaf of two owes that leaf's
    bytes at that leaf's level and nothing at the other's.
    """
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    if len(charged) != len(operands):  # pragma: no cover - internal caller contract
        raise AnalysisError("cost movement needs one charge per operand")
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    for moved, split in zip(cost.traffic, charged):
        for level, value in split.items():
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
    return levels, tuple(cost.traffic)


def _stated_movement(
    call: Call, cost: Cost, ctx: CostContext
) -> tuple[tuple[TrafficBytes, ...], tuple[dict[str, int], ...]]:
    """Per-operand movement as the Op's own access relation states it.

    A Type says how big a value is, not how much of it this occurrence touches.
    The relation is asked instead, in this context's window, so one handler
    answers for the whole program and for one unit. Only the amount comes from
    it: which direction an operand moves stays the cost's answer. Each amount
    comes back twice, as the operand's own total and split by the level of every
    leaf it reached, so one leaf of a mixed tuple owes bytes at its own level
    alone. A boundary nothing can charge in bytes is refused.
    """
    relations = relations_of(call, ctx)
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    stated: list[TrafficBytes] = []
    charged: list[dict[str, int]] = []
    for index, (operand, moved) in enumerate(zip(operands, cost.traffic)):
        if index == len(call.args):
            answer = _output_bytes(relations, ctx.local_type_of(call), umat_level=None)
        else:
            answer = _moved_bytes(
                ctx.local_type_of(operand),
                relations.inputs[index].pattern,
                umat_level=_UMAT_CONSUMPTION_LEVEL,
            )
        if answer is None:
            raise AnalysisError(
                f"{describe(call)}: boundary {index} reaches coordinates nothing "
                "here can charge in bytes"
            )
        moving, by_level = answer
        stated.append(
            TrafficBytes(moving if moved.read else 0, moving if moved.write else 0)
        )
        charged.append(by_level)
    return tuple(stated), tuple(charged)


def _output_bytes(
    relations: AccessRelations, held: Type, *, umat_level: str | None
) -> tuple[int, dict[str, int]] | None:
    """Bytes the result moves, taking one output boundary per field it has.

    A tuple result is as many boundaries as it has fields, each somewhere of its
    own that the Op stated separately. Reading only the first would drop the
    rest, and reading the tuple as one value has no element count to read at
    all -- either way the relation's answer is replaced by the Type's size.
    """
    fields = held.fields if isinstance(held, TupleType) else (held,)
    total = 0
    by_level: dict[str, int] = {}
    for position, field_ in enumerate(fields):
        if not 0 <= position < len(relations.outputs):
            return None
        answer = _moved_bytes(
            field_, relations.outputs[position].pattern, umat_level=umat_level
        )
        if answer is None:
            return None
        moving, levels = answer
        total += moving
        for level, value in levels.items():
            by_level[level] = by_level.get(level, 0) + value
    return total, by_level


def _moved_bytes(
    held: Type, pattern, *, umat_level: str | None
) -> tuple[int, dict[str, int]] | None:
    """The bytes one boundary moves of *held*, by the leaves it reaches.

    A structured operand is indexed by leaf and its leaves need not be the same
    width or live at the same level, so which ones a boundary reaches is what
    decides the bytes and where they are charged: charging the first for the one
    that was taken is a wrong number at the right size. A single leaf is counted
    in its own elements instead.
    """
    leaves = leaves_of(held)
    if not leaves:
        return None
    if len(leaves) == 1:
        size = _bytes_for(leaves[0], reached_elements(pattern))
        if size is None:
            return None
        return _charged(((leaves[0], size),), umat_level)
    reached = reached_leaves(pattern, len(leaves))
    if reached is None:
        return None
    taken: list[tuple[TensorType, int]] = []
    for leaf in sorted(reached):
        size = static_bytes(leaves[leaf])
        if size is None:
            return None
        taken.append((leaves[leaf], size))
    return _charged(tuple(taken), umat_level)


def _charged(
    taken: tuple[tuple[TensorType, int], ...], umat_level: str | None
) -> tuple[int, dict[str, int]]:
    """One boundary's bytes, and the ones a level was named for.

    An unmaterialized leaf has no residency of its own, so it is part of what
    moved without being part of any level's traffic unless the caller says where
    this occurrence materializes it.
    """
    total = 0
    by_level: dict[str, int] = {}
    for leaf, size in taken:
        total += size
        level = umat_level if leaf.storage is StorageKind.UMAT else str(leaf.storage)
        if level is None:
            continue
        by_level[level] = by_level.get(level, 0) + size
    return total, by_level


def _bytes_for(held: Type, elements: int | None) -> int | None:
    """The bytes *elements* of *held* occupy, or ``None`` when unanswerable.

    Counted from how wide one element is, rounded up: a packed dtype has no
    whole number of bytes per element, so a share of the whole would round one
    bool to nothing and nine of them to one byte instead of two. A value whose
    element width nobody states is not one this can answer for.
    """
    if elements is None or not isinstance(held, TensorType):
        return None
    bits = getattr(held.dtype, "bit_width", None)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        return None
    return -(-elements * bits // 8)


























def _as_related(
    expr: Call, cost: Cost, ctx: CostContext, types: tuple[Type, ...]
) -> tuple[Cost, tuple[dict[str, int], ...]]:
    """One cost with every amount taken from the Op's own relations.

    Asked in the window the caller is asking about, because one handler answers
    for the whole program and for one participant. A Function is not an Op with
    boundaries of its own -- what it moves is what its body does, charged at the
    levels its operands name here -- and every other target states its
    coordinates or is refused.
    """
    if isinstance(expr.target, Function):
        return cost, tuple(
            bytes_by_storage(
                type_,
                umat_level=_UMAT_CONSUMPTION_LEVEL if index < len(expr.args) else None,
            )
            for index, type_ in enumerate(types)
        )
    if access_relation_registry.lookup(type(expr.target)) is None:
        raise AnalysisError(
            f"{describe(expr)}: states no access relations, so nothing here says "
            "what it moves"
        )
    stated, charged = _stated_movement(expr, cost, ctx)
    return Cost(cost.flops, stated, cost.service), charged


def call_traffic(
    expr: Call, whole: CostEvaluator, local: CostEvaluator
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    The Type of the leaf it reached names the level those bytes are charged at,
    and an allocation does not correct either answer.
    """
    try:
        whole_cost = whole.visit(expr)
        local_cost = local.visit(expr)
    except (ValueError, VerifyError) as error:
        raise AnalysisError(str(error)) from None
    whole_types = (
        *(whole.ctx.local_type_of(arg) for arg in expr.args),
        whole.ctx.local_type_of(expr),
    )
    local_types = (
        *(local.ctx.local_type_of(arg) for arg in expr.args),
        local.ctx.local_output_type(expr),
    )
    whole_cost, whole_charged = _as_related(expr, whole_cost, whole.ctx, whole_types)
    local_cost, local_charged = _as_related(expr, local_cost, local.ctx, local_types)
    traffic_by_level, operands = _call_movement(expr, whole_cost, whole_charged)
    local_traffic, _local_operands = _call_movement(expr, local_cost, local_charged)
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
