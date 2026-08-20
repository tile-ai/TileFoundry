"""What each occurrence moves.

The movement half of an Op's registered cost: how much crossed each boundary,
which is what that boundary's own relation reaches, and at which storage level
those bytes are charged, which is a fact about the operand's Type.
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
from .walk import bytes_by_storage, describe, tensor_types

_UMAT_CONSUMPTION_LEVEL = str(StorageKind.RMEM)


def _call_movement(
    call: Call,
    cost: Cost,
    operand_types: tuple[Type, ...] | None = None,
) -> tuple[tuple[tuple[str, TrafficBytes], ...], tuple[TrafficBytes, ...]]:
    """What each operand of *call* moves, and where those bytes are charged.

    How much moves is what that boundary's own relation reaches; which level it
    moves at is a function of that operand's Type. *operand_types* supplies the
    projected types for the per-unit reading.
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


def _stated_movement(call: Call, cost: Cost, ctx: CostContext) -> tuple[TrafficBytes, ...]:
    """Per-operand movement as the Op's own access relation states it.

    A Type says how big a value is, not how much of it this occurrence touches.
    The relation is asked instead, in this context's window, so one handler
    answers for the whole program and for one unit. Only the amount comes from
    it: which direction an operand moves stays the cost's answer. A boundary
    nothing can charge in bytes is refused rather than answered from the Type it
    was written against.
    """
    relations = relations_of(call, ctx)
    operands = (*call.args, call)
    if len(cost.traffic) != len(operands):
        raise AnalysisError(
            f"{describe(call)}: cost reports {len(cost.traffic)} operands, "
            f"the call has {len(operands)}"
        )
    stated: list[TrafficBytes] = []
    for index, (operand, moved) in enumerate(zip(operands, cost.traffic)):
        if index == len(call.args):
            moving = _output_bytes(relations, ctx.local_type_of(call))
        else:
            moving = _moved_bytes(
                ctx.local_type_of(operand), relations.inputs[index].pattern
            )
        if moving is None:
            raise AnalysisError(
                f"{describe(call)}: boundary {index} reaches coordinates nothing "
                "here can charge in bytes"
            )
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
        if not 0 <= position < len(relations.outputs):
            return None
        moving = _moved_bytes(field_, relations.outputs[position].pattern)
        if moving is None:
            return None
        total += moving
    return total


def _moved_bytes(held: Type, pattern) -> int | None:
    """The bytes one boundary moves of *held*, by the leaves it reaches.

    A structured operand is indexed by leaf and its leaves need not be the same
    width, so which ones a boundary reaches is what decides the bytes: charging
    the first for the one that was taken is a wrong number at the right size. A
    single leaf is counted in its own elements instead.
    """
    leaves = leaves_of(held)
    if not leaves:
        return None
    if len(leaves) == 1:
        return _bytes_for(leaves[0], reached_elements(pattern))
    reached = reached_leaves(pattern, len(leaves))
    if reached is None:
        return None
    charged = 0
    for leaf in sorted(reached):
        size = static_bytes(leaves[leaf])
        if size is None:
            return None
        charged += size
    return charged


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


























def _as_related(expr: Call, cost: Cost, ctx: CostContext) -> Cost:
    """One cost with every amount taken from the Op's own relations.

    Asked in the window the caller is asking about, because one handler answers
    for the whole program and for one participant. A Function is not an Op with
    boundaries of its own -- what it moves is what its body does -- and every
    other target states its coordinates or is refused here.
    """
    if isinstance(expr.target, Function):
        return cost
    if access_relation_registry.lookup(type(expr.target)) is None:
        raise AnalysisError(
            f"{describe(expr)}: states no access relations, so nothing here says "
            "what it moves"
        )
    return Cost(cost.flops, _stated_movement(expr, cost, ctx), cost.service)


def call_traffic(
    expr: Call, whole: CostEvaluator, local: CostEvaluator
) -> TrafficMetadata:
    """What one Call moves, whole and for one participant.

    The same registered evaluator the work half reads, projected onto its
    movement instead of its flops. The evaluator says which way each boundary
    moves and whether it materialises; how much crosses it is what that
    boundary's own relation reaches, in whichever window is being asked about.
    Where the bytes land is the allocation's answer rather than a correction to
    this one.
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
    whole_cost = _as_related(expr, whole_cost, whole.ctx)
    local_cost = _as_related(expr, local_cost, local.ctx)
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
