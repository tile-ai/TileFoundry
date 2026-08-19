"""Turn one occurrence's access relations into the bytes it moves.

One lowering answers twice: for the program, and for one unit of it. A unit's
answer is the relation asked of a unit, already the projection onto one
participant -- the program's window is never intersected to arrive at it. The
plan is for the buffer: which one a boundary is in, where a unit's coordinates
sit in it, and whether a link's two sides are the same bytes. An access this
cannot place is refused rather than reported as nothing, a buffer with no
address and a unit moving what it holds no part of alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.shard import try_c_order_strides
from tilefoundry.ir.types.shard.layout import ComposedLayout
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
    """One element of *type_* in bytes, refused when it is not whole bytes."""
    bits = getattr(getattr(type_, "dtype", None), "bit_width", None)
    if not isinstance(bits, int) or bits <= 0 or bits % 8:
        raise AnalysisError(
            f"an element of {bits!r} bits has no whole number of bytes to charge"
        )
    return bits // 8


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
    which level it crossed at is where the value lives. A boundary whose value
    has no address is refused, an occurrence that cannot be located not having
    been shown to move nothing, and one that moved nothing gets no row at all.
    One participant is asked: every participant holds the same extents of a
    value and differs only in where they start.
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
    settled = _settled(unit_relations, plan, call, unit_domain, unit_ctx)
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
            if not plan.of(value):
                raise AnalysisError(
                    f"{where} names a value with no address, so the bytes it "
                    "moves cannot be attributed to a buffer"
                )
            wanted = 0 if field is None else field
            owned = plan.owned(value, participant, wanted) if runs else None
            if owned is None:
                if share:
                    raise AnalysisError(
                        f"{where} moves {share} elements of a value this "
                        "participant holds no part of, so the bytes come from "
                        "somewhere this does not model"
                    )
            else:
                _within(mine, owned, unit_domain, where)
        payload = _element_bytes(leaf)
        read, write = (moving, 0) if side == "input" else (0, moving)
        boundaries.append(
            BoundaryTraffic(
                side=side,
                index=index,
                field=field,
                level=level,
                read=read * payload,
                write=write * payload,
            )
        )
        into = whole.setdefault(level, [0, 0])
        into[0] += read * payload
        into[1] += write * payload
        held = unit.setdefault(level, [0, 0])
        held[0] += (share if side == "input" else 0) * payload
        held[1] += (0 if side == "input" else share) * payload
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
    domain: tuple,
    ctx,
) -> "set[tuple[str, int]]":
    """The boundaries whose bytes were already where the operation put them.

    A read or a write charges what it moved whether or not the bytes were
    already where they ended up: an operation that read an operand read it. A
    transfer is the one thing that can come to nothing, and only when every link
    it is made of names the same addresses on both sides. This is asked of one
    unit's relations, because those state their coordinates in the positions the
    buffers are addressed by.
    """
    proven: set[tuple[str, int]] = set()
    sides = [("input", index, item) for index, item in enumerate(relations.inputs)]
    sides += [("output", index, item) for index, item in enumerate(relations.outputs)]
    for side, index, boundary in sides:
        if boundary.mode is not AccessMode.TRANSFER or boundary.quantity.upper == 0:
            continue
        links = _links_for(side, boundary, relations)
        if links and all(
            _same_address(link, output, call, plan, domain, ctx) for link, output in links
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


def _same_address(link, field: int | None, call, plan: BufferPlan, domain: tuple, ctx) -> bool:
    """Whether a link's two sides name the same bytes.

    Same buffer is necessary and not sufficient: two regions of one allocation
    are different bytes unless the coordinates map to the same addresses. A pair
    of windows written the same way over the same layout answers that without
    knowing where they sit, which is what lets a run-time offset be honoured.
    """
    source = _entry(plan, call.args[link.input], link.input_field)
    output = _entry(plan, call, field)
    if source is None or output is None:
        return False
    if (source.ref.buffer_id, source.ref.level) != (output.ref.buffer_id, output.ref.level):
        return False
    where = describe(call)
    reading = _widths(link, field, call, ctx, where)
    if reading is None:
        return False
    source_bytes, output_bytes = reading
    if isinstance(link.source, WindowAccess) or isinstance(link.output, WindowAccess):
        return link.source == link.output and _seat(source.ref, source_bytes) == _seat(
            output.ref, output_bytes
        )
    left = _address_map(link.source, source.ref, source_bytes, domain)
    right = _address_map(link.output, output.ref, output_bytes, domain)
    return left is not None and right is not None and left.is_equal(right)


def _seat(ref, payload: int) -> "tuple | None":
    """Where one value sits in its buffer and how its coordinates step there.

    Two values of one buffer name the same bytes only if they start at the same
    byte and step the same way. Either being unreadable is not a match, because
    an address nobody can state has not been shown to equal another.
    """
    strides = _strides(ref)
    seated = _seated(ref)
    if strides is None or seated is None:
        return None
    return (ref.offset + seated * payload, tuple(stride * payload for stride in strides))


def _widths(link, field: int | None, call, ctx, where: str) -> "tuple[int, int] | None":
    """What one element costs on each side of a link, in bytes."""
    try:
        source = _element_bytes(_leaf(ctx.type_of(call.args[link.input]), link.input_field, where))
        output = _element_bytes(_leaf(ctx.type_of(call), field, where))
    except AnalysisError:
        return None
    return source, output


def _entry(plan: BufferPlan, value, field: int | None) -> PlannedBuffer | None:
    """One planned field of a value, or None when it has none."""
    wanted = 0 if field is None else field
    return next((item for item in plan.of(value) if item.field == wanted), None)


def _strides(ref) -> tuple | None:
    """One buffer's element strides, in the positions it is addressed by.

    A layout wrapped in a distribution or shifted by a constant still steps the
    way the layout underneath it steps, so the wrappers are unwrapped rather
    than read for strides they do not carry.

    A value that states no layout at all is dense in its own coordinates, which
    is what the rest of the compiler reads it as, so it steps in C order over
    the extents it has. Extents nobody has bound yet give no such order, and
    those stay unknown rather than guessed.
    """
    return _stepping(ref.layout, ref.shape)


def _stepping(layout, shape: tuple = ()) -> tuple | None:
    """The element strides of whatever layout finally states them."""
    if layout is None:
        stated = try_c_order_strides(tuple(shape))
        return None if stated is None else tuple(stated)
    if isinstance(layout, ComposedLayout):
        return None if layout.inner is not None else _stepping(layout.outer)
    inner = getattr(layout, "layout", None)
    if inner is not None:
        return _stepping(inner)
    strides = getattr(layout, "strides", None)
    if strides is None:
        return None
    return (
        tuple(strides)
        if all(isinstance(item, int) and not isinstance(item, bool) for item in strides)
        else None
    )


def _seated(ref) -> "int | None":
    """How many elements into its buffer one value's own coordinates start.

    A layout composed with a constant carries that shift, and a shift is part of
    an address: two views of one buffer that differ only by it are different
    bytes. A composition this cannot read is not a zero shift, so it answers
    with nothing rather than with the front of the buffer.
    """
    return _shift(ref.layout)


def _shift(layout) -> "int | None":
    """Every constant shift between one value's coordinates and its buffer."""
    if layout is None:
        return 0
    if isinstance(layout, ComposedLayout):
        if layout.inner is not None or not isinstance(layout.offset, int):
            return None
        deeper = _shift(layout.outer)
        return None if deeper is None else layout.offset + deeper
    inner = getattr(layout, "layout", None)
    return 0 if inner is None else _shift(inner)


def _address_map(pattern, ref, payload: int, domain: tuple) -> "isl.map | None":
    """Where a pattern's coordinates land, as byte addresses in one buffer.

    In bytes and from the buffer's own front, because two values of one buffer
    at different byte offsets are different bytes however alike their
    coordinates read. Over the iteration the occurrence runs, and not over every
    coordinate the formula could take: a position a participant holds one of
    contributes nothing to an address, and two patterns that differ only there
    name the same bytes.
    """
    seat = _seat(ref, payload)
    if seat is None or isinstance(pattern, (IndexedAccess, WindowAccess)):
        return None
    base, strides = seat
    coords = ", ".join(f"c{axis}" for axis in range(len(strides)))
    terms = [f"{stride}*c{axis}" for axis, stride in enumerate(strides) if stride]
    linear = " + ".join([*terms, str(base)]) if terms else str(base)
    try:
        relation = pattern.as_map() if hasattr(pattern, "as_map") else pattern
        placed = relation.apply_range(isl.map(f"{{ [{coords}] -> [{linear}] }}"))
        return placed.intersect_domain(_box(domain))
    except (isl.Error, ValueError):
        return None


def view_displacement(link, source, source_bytes: int, output, output_bytes: int, domain: tuple):
    """Where in its source a renamed value begins, in bytes, when that is fixed.

    A link's two sides read one iteration domain, so the distance between the
    bytes they name is a function of that domain. A renaming is the case where
    that distance does not vary: the value sits at one place in the buffer, and
    the place is the distance. One that varies is not a displacement of the
    value but a mapping through it, and one side that cannot be addressed at all
    -- a window whose start is a run-time value -- fixes nothing.
    """
    reading = _address_map(link.source, source, source_bytes, domain)
    written = _address_map(link.output, output, output_bytes, domain)
    if reading is None or written is None:
        return None
    try:
        apart = (
            reading.flat_range_product(written)
            .apply_range(isl.map("{ [a, b] -> [a - b] }"))
            .range()
        )
        if not apart.is_singleton():
            return None
        return int(str(apart.max_val(isl.aff("{ [x] -> [x] }"))))
    except (isl.Error, ValueError):
        return None


__all__ = ["BoundaryTraffic", "TrafficMetadata", "lower_traffic", "view_displacement"]
