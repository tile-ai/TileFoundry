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


def _reached_bytes(
    boundaries: tuple[tuple[Type, object], ...], umat_level: str | None
) -> tuple[int, dict[str, int]] | None:
    """What one operand's boundaries reach, in bytes and per level.

    A structured operand is indexed by leaf and its leaves need not be the same
    width or live at the same level, so which ones a boundary reaches decides
    both numbers: charging the first for the one that was taken is a wrong
    number at the right size. A single leaf is counted in its own elements
    instead. A leaf nobody materialised is part of what moved and part of no
    level's traffic unless the caller says where this occurrence puts it.
    """
    total = 0
    by_level: dict[str, int] = {}
    for held, pattern in boundaries:
        leaves = leaves_of(held)
        if not leaves:
            return None
        if len(leaves) == 1:
            taken = {0: _bytes_for(leaves[0], reached_elements(pattern))}
        else:
            reached = reached_leaves(pattern, len(leaves))
            if reached is None:
                return None
            taken = {index: static_bytes(leaves[index]) for index in sorted(reached)}
        for index, size in taken.items():
            if size is None:
                return None
            total += size
            leaf = leaves[index]
            level = umat_level if leaf.storage is StorageKind.UMAT else str(leaf.storage)
            if level is not None:
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


def _movement(
    call: Call, cost: Cost, ctx: CostContext, types: tuple[Type, ...]
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and the levels those bytes are at.

    A Type says how big a value is, not how much of it this occurrence touches,
    so the amount is what that boundary's relation reaches in this context's
    window -- one handler answering for the whole program and for one unit. The
    direction stays the cost's answer and the level is where the reached leaf
    lives, so one leaf of two owes its own bytes at its own level. A Function has
    no boundaries of its own, and every other target states its coordinates or
    is refused, as is a boundary nothing can charge in bytes.
    """
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    stated = cost.traffic
    if isinstance(call.target, Function):
        charged = [
            bytes_by_storage(
                type_,
                umat_level=_UMAT_CONSUMPTION_LEVEL if index < len(call.args) else None,
            )
            for index, type_ in enumerate(types)
        ]
    else:
        if access_relation_registry.lookup(type(call.target)) is None:
            raise AnalysisError(
                f"{describe(call)}: states no access relations, so nothing here "
                "says what it moves"
            )
        relations = relations_of(call, ctx)
        result = ctx.local_type_of(call)
        fields = result.fields if isinstance(result, TupleType) else (result,)
        if len(fields) > len(relations.outputs):
            raise AnalysisError(
                f"{describe(call)}: states {len(relations.outputs)} output "
                f"boundaries for a result of {len(fields)} fields"
            )
        amounts, charged = [], []
        for index, moved in enumerate(cost.traffic):
            if index == len(call.args):
                asked = tuple(
                    (field_, relations.outputs[position].pattern)
                    for position, field_ in enumerate(fields)
                )
                level = None
            else:
                asked = (
                    (types[index], relations.inputs[index].pattern),
                )
                level = _UMAT_CONSUMPTION_LEVEL
            answer = _reached_bytes(asked, level)
            if answer is None:
                raise AnalysisError(
                    f"{describe(call)}: boundary {index} reaches coordinates "
                    "nothing here can charge in bytes"
                )
            moving, by_level = answer
            amounts.append(
                TrafficBytes(moving if moved.read else 0, moving if moved.write else 0)
            )
            charged.append(by_level)
        stated = tuple(amounts)
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    for moved, split in zip(stated, charged):
        for level, value in split.items():
            if moved.read:
                reads[level] = reads.get(level, 0) + value
            if moved.write:
                writes[level] = writes.get(level, 0) + value
    moving = any(item.read or item.write for item in stated)
    levels = (
        tuple(
            (level, TrafficBytes(reads.get(level, 0), writes.get(level, 0)))
            for level in sorted(set(reads) | set(writes))
        )
        if moving
        else ()
    )
    return levels, stated


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
    asked = []
    for evaluator, cost in ((whole, whole_cost), (local, local_cost)):
        types = (
            *(evaluator.ctx.local_type_of(arg) for arg in expr.args),
            evaluator.ctx.local_type_of(expr),
        )
        asked.append(_movement(expr, cost, evaluator.ctx, types))
    (whole_levels, operands), (unit_levels, _unit_operands) = asked
    return TrafficMetadata(
        whole=whole_levels, per_unit=unit_levels, operands=operands
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
