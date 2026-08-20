"""What each occurrence moves.

The movement half of an Op's registered cost: how much crossed each boundary,
which is what that boundary's own relation reaches, and at which storage level
those bytes are charged, which is a fact about the operand's Type.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_elements,
    access_relation_registry,
    elements_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .errors import AnalysisError
from .metadata import TrafficBytes, TrafficMetadata
from .walk import bytes_by_storage, describe, tensor_types

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
    if access_relation_registry.lookup(type(call.target)) is None:
        return None
    relations = relations_of(call, ctx)
    operands = (*call.args, call)
    stated: list[TrafficBytes] = []
    for index, (operand, moved) in enumerate(zip(operands, cost.traffic)):
        if index == len(call.args):
            moving = _output_bytes(relations, ctx.local_type_of(call))
        else:
            moving = _bytes_for(
                ctx.local_type_of(operand),
                access_elements(relations, boundary=index),
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
        moving = _bytes_for(
            field_, access_elements(relations, boundary=position, output=True)
        )
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


























def call_traffic(
    expr: Call, whole: CostEvaluator, local: CostEvaluator
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. What the Op states is what is charged: an
    operation that computed something computed it, and where the bytes land is
    the allocation's answer rather than a correction to this one.
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
