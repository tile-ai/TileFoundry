"""Turn one occurrence's access relations into the bytes it moves.

One lowering answers twice: for the program, and for one unit of it. A unit's
answer is the relation asked of a unit, already the projection onto one
participant -- the program's window is never intersected to arrive at it. The
plan is for the buffer: which one a boundary is in, where a unit's coordinates
sit in it, and whether a link is the same bytes on both sides. It says what can
be proved, not what may be charged: an access it cannot place is charged for the
copy nobody ruled out, and only a unit reading what another holds is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    AccessMode,
    AccessRelations,
    BoundaryAccess,
    IndexedAccess,
    OperandValue,
    WindowAccess,
)
from tilefoundry.visitor_registry.contexts import TrafficBytes

from .buffer_plan import BufferPlan, PlannedBuffer
from .errors import AnalysisError
from .walk import describe, tensor_types


@dataclass(frozen=True)
class BoundaryTraffic:
    """What one boundary of one occurrence moves, and where those bytes are."""

    side: str
    index: int
    field: int | None
    level: str
    read: int = 0
    write: int = 0

    def __post_init__(self) -> None:
        if self.side not in ("input", "output"):
            raise ValueError(f"a boundary is an input or an output, not {self.side!r}")
        for name in ("read", "write"):
            moved = getattr(self, name)
            if not isinstance(moved, int) or isinstance(moved, bool) or moved < 0:
                raise ValueError(f"a boundary moves a whole number of bytes, not {moved!r}")


@dataclass(frozen=True)
class TrafficMetadata(IRMetadata):
    """The bytes one occurrence moves, whole and for one participant.

    ``boundaries`` states the whole program's movement one boundary at a time,
    each tuple field of its own, so a reader can see which operand a level's
    bytes came from rather than a total it has to take on trust.
    """

    whole: tuple[tuple[str, TrafficBytes], ...] = ()
    per_unit: tuple[tuple[str, TrafficBytes], ...] = ()
    boundaries: tuple[BoundaryTraffic, ...] = ()


def _element_bytes(type_: Type) -> int:
    """One element of *type_* in bytes, refused when it is not whole bytes.

    An address is a byte, so a value whose elements are not whole bytes has no
    address arithmetic here. How much such a value moves is a different question
    and is answered over the whole count rather than one element at a time.
    """
    bits = getattr(getattr(type_, "dtype", None), "bit_width", None)
    if not isinstance(bits, int) or bits <= 0 or bits % 8:
        raise AnalysisError(
            f"an element of {bits!r} bits has no whole number of bytes to charge"
        )
    return bits // 8


def _moved_bytes(count: int, type_: Type) -> int:
    """What *count* elements of *type_* come to in bytes, rounded up as a leaf.

    A leaf is addressed on its own, so a value of sub-byte elements takes whole
    bytes however few of them move -- which is what the type system already
    answers when asked the size of one.
    """
    bits = getattr(getattr(type_, "dtype", None), "bit_width", None)
    if not isinstance(bits, int) or bits <= 0:
        raise AnalysisError(f"an element of {bits!r} bits has no size to charge")
    return -(-count * bits // 8)


def _leaf(type_: Type, field: int | None, where: str) -> TensorType:
    """The tensor a boundary names, which is one field of a tuple or the value.

    A boundary over a whole tuple is charged as its fields, which needs them to
    agree about what a field costs and where it is. Fields that do not agree are
    two questions wearing one boundary, and are refused rather than charged at
    whichever one comes first.
    """
    if not isinstance(type_, TupleType):
        if not isinstance(type_, TensorType):
            raise AnalysisError(f"{where} names a tensor, and this is {type_!r}")
        return type_
    fields = tensor_types(type_)
    if field is not None:
        if not 0 <= field < len(fields):
            raise AnalysisError(f"{where} names field {field} of {len(fields)}")
        return fields[field]
    if not fields:
        raise AnalysisError(f"{where} names a tuple with no tensor in it")
    kinds = {(leaf.dtype, leaf.storage) for leaf in fields}
    if len(kinds) != 1:
        raise AnalysisError(
            f"{where} names a whole tuple whose fields are stored and sized "
            "differently, so one row cannot say what it moved"
        )
    return fields[0]


def _box(shape: tuple) -> isl.set:
    """Every coordinate of a value with these extents."""
    if not shape:
        return isl.set("{ [] }")
    names = [f"i{axis}" for axis in range(len(shape))]
    guards = " and ".join(
        f"0 <= {name} < {extent}" for name, extent in zip(names, shape)
    )
    return isl.set(f"{{ [{', '.join(names)}] : {guards} }}")


def _window_set(window: WindowAccess, shape: tuple) -> isl.set | None:
    """The coordinates a window covers, or None when only run time knows.

    A complement is the same window read from the other side, so it is taken as
    what the container has left once the window is removed.
    """
    if any(isinstance(edge, OperandValue) for edge in (*window.offsets, *window.extents)):
        return None
    if any(not isinstance(edge, int) for edge in (*window.offsets, *window.extents)):
        return None
    names = [f"i{axis}" for axis in range(len(shape))]
    if not names:
        return isl.set("{ [] }")
    guards = " and ".join(
        f"{offset} <= {name} < {offset + extent}"
        for name, offset, extent in zip(names, window.offsets, window.extents)
    )
    covered = isl.set(f"{{ [{', '.join(names)}] : {guards} }}")
    whole = _box(shape)
    return whole.subtract(covered) if window.complement else covered.intersect(whole)


def _touched(pattern, shape: tuple, domain: tuple) -> isl.set | None:
    """Which coordinates of a boundary's value one occurrence reaches.

    An affine carrier is read through the iteration it runs over; a window says
    its own coordinates outright. A lookup names coordinates a run-time index
    chooses, so it reaches nothing this can write down.
    """
    if isinstance(pattern, IndexedAccess):
        return None
    if isinstance(pattern, WindowAccess):
        return _window_set(pattern, shape)
    try:
        relation = pattern.as_map() if hasattr(pattern, "as_map") else pattern
        return relation.intersect_domain(_box(domain)).range()
    except (isl.Error, ValueError):
        return None


def _count(region: isl.set) -> int | None:
    """How many coordinates a region holds, when it holds a countable number."""
    try:
        value = region.count_val()
    except (isl.Error, ValueError):
        return None
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else None


def _translated(region: isl.set, origin: tuple) -> isl.set:
    """One participant's own coordinates, said in the whole buffer's."""
    if not origin:
        return region
    names = [f"i{axis}" for axis in range(len(origin))]
    image = ", ".join(
        name if not start else f"{name} + {start}" for name, start in zip(names, origin)
    )
    return region.apply(isl.map(f"{{ [{', '.join(names)}] -> [{image}] }}"))


def _within(
    boundary: BoundaryAccess,
    owned: PlannedBuffer,
    domain: tuple,
    where: str,
) -> None:
    """Hold one unit's access to coordinates its buffer is addressed by.

    A unit states its access in its own coordinates, and the origin the plan
    gives it says where those sit in the buffer. How far it reaches is not
    checked here: a participant that gathers reads what its neighbours hold, and
    at a level held per participant those bytes are the same buffer at another
    position. How many coordinates it reaches by is, because an access of a
    different rank than the buffer is not an access to it.
    """
    reached = _touched(boundary.pattern, tuple(owned.extents), domain)
    if reached is None:
        return
    if reached.dim(isl.dim_type.SET) != len(owned.origin):
        raise AnalysisError(
            f"{where} reaches {reached.dim(isl.dim_type.SET)} coordinates of a "
            f"buffer addressed by {len(owned.origin)}, so where it lands in that "
            "buffer cannot be said"
        )


def _rows_of(
    call,
    relations: AccessRelations,
    ctx,
) -> "list[tuple[str, int, int | None, BoundaryAccess, object]]":
    """Every boundary of one occurrence, with the value and field it names."""
    result = ctx.type_of(call)
    outputs = result.fields if isinstance(result, TupleType) else (result,)
    rows = [
        ("input", index, _input_field(relations, index), boundary, call.args[index])
        for index, boundary in enumerate(relations.inputs)
    ]
    rows += [
        ("output", index, index if len(outputs) > 1 else None, boundary, call)
        for index, boundary in enumerate(relations.outputs)
    ]
    return rows


def _input_field(relations: AccessRelations, operand: int) -> int | None:
    """Which field of a tuple operand a boundary is about, when it says so.

    A link that forwards one field of a tuple names it, and that is the field
    the boundary moved. Links that name different fields of one operand leave
    the boundary about the operand rather than about any one field.
    """
    named = {
        link.input_field
        for output in relations.outputs
        for link in (output.storage.links if output.storage else ())
        if link.input == operand and link.input_field is not None
    }
    return named.pop() if len(named) == 1 else None


def lower_traffic(
    call,
    relations: AccessRelations,
    unit_relations: AccessRelations,
    plan: BufferPlan,
    ctx,
    unit_ctx,
    *,
    participant: int,
    runs: bool = True,
    umat_level: str,
) -> TrafficMetadata:
    """The bytes one occurrence moves, whole and for one participant.

    Both readings come from the same boundaries, one asked of the program and
    one of a unit of it. How much crossed a boundary is the Op's own answer, and
    which level it crossed at is where the value lives. What proves a transfer
    came to nothing is the placement having put its output in the allocation the
    link names, so a boundary the plan places nowhere is charged for the copy
    nothing ruled out. Refused outright is a unit running an occurrence while
    holding no part of a value the plan did place.
    """
    unit_domain = tuple(getattr(unit_ctx.local_type_of(call), "shape", ()) or ())
    boundaries: list[BoundaryTraffic] = []
    whole: dict[str, list[int]] = {}
    unit: dict[str, list[int]] = {}
    rows = _rows_of(call, relations, ctx)
    unit_rows = {
        (side, index): boundary
        for side, index, _field, boundary, _value in _rows_of(call, unit_relations, ctx)
    }
    settled = _settled(unit_relations, plan, call)
    for side, index, field, boundary, value in rows:
        where = f"{describe(call)}: {side} {index}"
        mine = unit_rows.get((side, index))
        if mine is None:
            raise AnalysisError(f"{where} is stated for the program and not for a unit")
        moving = 0 if (side, index) in settled else boundary.quantity.upper
        share = 0 if moving == 0 or not runs else mine.quantity.upper
        if not moving and not share:
            continue
        leaf = _leaf(ctx.type_of(value), field, where)
        if leaf.storage is StorageKind.UMAT:
            if side == "output":
                continue
            level = umat_level
        else:
            level = str(leaf.storage)
            wanted = 0 if field is None else field
            owned = plan.owned(value, participant, wanted) if runs else None
            if owned is not None:
                _within(mine, owned, unit_domain, where)
            elif runs and share and plan.of(value):
                raise AnalysisError(
                    f"{where} moves {share} elements of a value this "
                    "participant holds no part of, so the bytes come from "
                    "somewhere this does not model"
                )
        read, write = (moving, 0) if side == "input" else (0, moving)
        crossed = (_moved_bytes(read, leaf), _moved_bytes(write, leaf))
        boundaries.append(
            BoundaryTraffic(
                side=side,
                index=index,
                field=field,
                level=level,
                read=crossed[0],
                write=crossed[1],
            )
        )
        into = whole.setdefault(level, [0, 0])
        into[0] += crossed[0]
        into[1] += crossed[1]
        held = unit.setdefault(level, [0, 0])
        held[0] += _moved_bytes(share if side == "input" else 0, leaf)
        held[1] += _moved_bytes(0 if side == "input" else share, leaf)
    return TrafficMetadata(
        whole=_levelled(whole),
        per_unit=_levelled(unit),
        boundaries=tuple(boundaries),
    )


def _levelled(found: dict[str, list[int]]) -> tuple[tuple[str, TrafficBytes], ...]:
    """Level totals, in a stable order, leaving out levels nothing crossed."""
    return tuple(
        (level, TrafficBytes(read, write))
        for level, (read, write) in sorted(found.items())
        if read or write
    )


def _settled(
    relations: AccessRelations,
    plan: BufferPlan,
    call,
) -> "set[tuple[str, int]]":
    """The boundaries whose bytes were already where the operation put them.

    A read or a write charges what it moved whether or not the bytes were
    already where they ended up: an operation that read an operand read it. A
    transfer is the one thing that can come to nothing, and only when every link
    it is made of put its output in the allocation that link names -- a boundary
    made of several links comes to nothing only if all of them did.
    """
    proven: set[tuple[str, int]] = set()
    sides = [("input", index, item) for index, item in enumerate(relations.inputs)]
    sides += [("output", index, item) for index, item in enumerate(relations.outputs)]
    for side, index, boundary in sides:
        if boundary.mode is not AccessMode.TRANSFER or boundary.quantity.upper == 0:
            continue
        links = _links_for(side, boundary, relations)
        if links and all(
            _lives_in_it(link, output, call, plan) for link, output in links
        ):
            proven.add((side, index))
    return proven


def _links_for(
    side: str, boundary: BoundaryAccess, relations: AccessRelations
) -> "list[tuple[object, int | None]]":
    """The links that decide one boundary, with the output field each lands in."""
    if side == "output":
        index = relations.outputs.index(boundary)
        field = index if len(relations.outputs) > 1 else None
        return [(link, field) for link in (boundary.storage.links if boundary.storage else ())]
    operand = relations.inputs.index(boundary)
    found = []
    for index, output in enumerate(relations.outputs):
        field = index if len(relations.outputs) > 1 else None
        for link in output.storage.links if output.storage else ():
            if link.input == operand:
                found.append((link, field))
    return found


def _lives_in_it(link, field: int | None, call, plan: BufferPlan) -> bool:
    """Whether this link's output was given the operand's allocation.

    The placement already answered this, and answered it once. A value whose
    operation was shown to forward or update another was never given a buffer
    of its own: it was given that other value's, and a value living in another's
    bytes did not move to get there. One that could not be shown to do so was
    allocated separately, and a link between two allocations is a copy.
    """
    source = _entry(plan, call.args[link.input], link.input_field)
    output = _entry(plan, call, field)
    if source is None or output is None:
        return False
    return (source.ref.buffer_id, source.ref.level) == (
        output.ref.buffer_id,
        output.ref.level,
    )


def _entry(plan: BufferPlan, value, field: int | None) -> PlannedBuffer | None:
    """One planned field of a value, or None when it has none."""
    wanted = 0 if field is None else field
    return next((item for item in plan.of(value) if item.field == wanted), None)


__all__ = ["BoundaryTraffic", "TrafficMetadata", "lower_traffic"]
