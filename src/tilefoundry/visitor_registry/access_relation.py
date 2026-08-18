"""Register what one operation reads, writes, and where its result's bytes live.

Boundary handlers return one ``BoundaryAccess`` per boundary: a pattern saying
where it reads -- isl where that is affine, ``IndexedAccess`` or
``WindowAccess`` where a runtime value decides it -- and, separately, how much
it moves. Neither is inferred from the other.

The same registration says whether the result owns its bytes or reuses an
operand's, as a claim about that Op's own types. Whoever walks the function
settles it; an identity relation alone claims nothing.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Union

import isl

from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of

from .registries import AnalysisRegistry


@dataclass(frozen=True)
class IndexedAccess:
    """One boundary read through the values of another operand.

    A gather names which source coordinate it wants by the element it reads out
    of ``index_operand``, so the address is not known here. Which operand is
    being indexed is said outright rather than assumed, because an Op may gather
    from more than one -- a rotation reads two tables through one set of
    positions.
    """

    source_operand: int
    index_operand: int
    source_axis: int


@dataclass(frozen=True)
class OperandValue:
    """One runtime number an access depends on, and what it may be.

    ``element`` picks a field when the operand is a tuple of scalars. ``bound``
    is the Op's own contract for the value, which is what lets a quantity stay a
    checkable range instead of widening to the whole operand.
    """

    operand: int
    element: int | None = None
    bound: tuple[int, int] | None = None


@dataclass(frozen=True)
class WindowAccess:
    """One boundary read or written as a window, one entry per axis.

    An ``offset`` says where the window starts on its axis and an ``extent`` how
    far it runs; either may be a number written down or an ``OperandValue`` only
    known at run time. Where the window sits never changes how much it covers,
    which is why an unbound offset costs a quantity nothing. ``complement``
    marks the part of the container the window leaves alone -- the same question
    answered from the other side.
    """

    offsets: tuple["OperandValue | int", ...]
    extents: tuple["OperandValue | object", ...]
    complement: bool = False


AccessPattern = Union["isl.multi_aff", "isl.map", IndexedAccess, WindowAccess]







@dataclass(frozen=True)
class AccessQuantity:
    """How many elements one boundary moves in one execution of an Op.

    ``lower`` and ``upper`` are equal whenever the Op can say the number. When
    they differ, ``provenance`` names the invariant the range came from, so a
    reader can check it rather than trust it; a prediction takes ``upper`` and
    says that it did.
    """

    lower: int
    upper: int
    provenance: str | None = None

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            edge = getattr(self, name)
            if not isinstance(edge, int) or isinstance(edge, bool) or edge < 0:
                raise ValueError(
                    f"an access moves a non-negative whole number of elements, "
                    f"and its {name} is {edge!r}"
                )
        if self.lower > self.upper:
            raise ValueError(
                f"an access of {self.lower}..{self.upper} elements runs backwards"
            )
        if self.lower != self.upper and not self.provenance:
            raise ValueError(
                f"an access of {self.lower}..{self.upper} elements is a range, so it "
                "must name the invariant it came from"
            )

    @property
    def exact(self) -> bool:
        """Whether one number answers this, rather than a range."""
        return self.lower == self.upper


class AccessMode(enum.Enum):
    """What a boundary is for, which is not the same as what it costs.

    ``READ`` and ``WRITE`` are the operation consuming or producing elements.
    They are charged whether or not the bytes turn out to be somewhere they
    already were: an add whose result lands in its operand's buffer still read
    that operand and still wrote its result. ``TRANSFER`` is an operation whose
    whole purpose is to move or re-address bytes, and is the only mode that can
    come to nothing -- when its links are shown to name the same addresses.
    """

    READ = "read"
    WRITE = "write"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class StorageLink:
    """One region an output may share with an input, and how much of it.

    ``source`` and ``output`` map one shared iteration domain to a coordinate on
    each side. Two patterns rather than two byte offsets, because an offset and
    a size cannot state an unbound window start or the mapping a transpose
    forwards through. ``quantity`` is how much the region holds, never measured
    off a pattern. A link is a candidate: whether these bytes really are shared
    is the allocation's answer, and an unhonoured link is the copy instead.
    """

    kind: str
    input: int
    source: "AccessPattern"
    output: "AccessPattern"
    quantity: "AccessQuantity"
    input_field: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("forward", "preserve"):
            raise ValueError(
                f"a link either forwards a value or preserves a container, "
                f"not {self.kind!r}"
            )
        if not isinstance(self.input, int) or isinstance(self.input, bool) or self.input < 0:
            raise ValueError(f"a link names an operand by position, not {self.input!r}")
        if self.input_field is not None and (
            not isinstance(self.input_field, int)
            or isinstance(self.input_field, bool)
            or self.input_field < 0
        ):
            raise ValueError(
                f"a link names a field of its operand by position, not "
                f"{self.input_field!r}"
            )
        for side, pattern in (("source", self.source), ("output", self.output)):
            if not isinstance(
                pattern, (isl.multi_aff, isl.map, IndexedAccess, WindowAccess)
            ):
                raise ValueError(
                    f"a link's {side} reads through a relation, a lookup or a "
                    f"window, not through {pattern!r}"
                )
        if not isinstance(self.quantity, AccessQuantity):
            raise ValueError(
                f"a link states how much it covers, not {self.quantity!r}"
            )


@dataclass(frozen=True)
class OutputStorage:
    """Where one output's bytes may already be. No links means fresh bytes."""

    links: tuple[StorageLink, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.links, tuple) or not all(
            isinstance(item, StorageLink) for item in self.links
        ):
            raise ValueError(f"storage is a tuple of links, not {self.links!r}")


@dataclass(frozen=True)
class BoundaryAccess:
    """One boundary: where it reads, and how much it moves.

    The two are separate answers. A pattern says which coordinates a value came
    from and is what a dependence needs; a quantity says how much crossed the
    boundary and is what a cost needs. Neither follows from the other -- every
    output of a scan depends on the whole input the scan reads once -- so the Op
    states both rather than having one inferred from the other.
    """

    pattern: AccessPattern
    quantity: AccessQuantity
    mode: AccessMode = AccessMode.READ
    storage: OutputStorage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AccessMode):
            raise ValueError(f"a boundary reads, writes or transfers, not {self.mode!r}")
        if self.storage is not None and not isinstance(self.storage, OutputStorage):
            raise ValueError(f"a boundary's storage is an OutputStorage, not {self.storage!r}")
        if not isinstance(
            self.pattern, (isl.multi_aff, isl.map, IndexedAccess, WindowAccess)
        ):
            raise ValueError(
                f"a boundary reads through a relation, a lookup or a window, "
                f"not through {self.pattern!r}"
            )
        if not isinstance(self.quantity, AccessQuantity):
            raise ValueError(
                f"a boundary states how much it moves, not {self.quantity!r}"
            )


def moves(pattern: "AccessPattern", count: int) -> BoundaryAccess:
    """One operand read, of a size the Op can state."""
    return BoundaryAccess(pattern, AccessQuantity(count, count), AccessMode.READ)


def writes(pattern: "AccessPattern", count: int) -> BoundaryAccess:
    """One result produced, of a size the Op can state.

    A write is charged wherever the bytes turn out to be. An operation that
    computed something computed it, and landing in an operand's buffer is a
    fact about the allocation rather than about the work.
    """
    return BoundaryAccess(pattern, AccessQuantity(count, count), AccessMode.WRITE)


def transfers(
    pattern: "AccessPattern", quantity: "AccessQuantity", *links: StorageLink
) -> BoundaryAccess:
    """One boundary whose whole purpose is moving or re-addressing bytes.

    The only mode that can come to nothing, and only when its links are shown
    to name the same addresses. It must name at least one, or there is nothing
    to compare and nothing said about where the bytes came from.
    """
    return BoundaryAccess(
        pattern, quantity, AccessMode.TRANSFER, OutputStorage(tuple(links))
    )


def moves_between(
    pattern: "AccessPattern", lower: int, upper: int, provenance: str
) -> BoundaryAccess:
    """One boundary whose amount an unbound value leaves within a range."""
    return BoundaryAccess(pattern, AccessQuantity(lower, upper, provenance))


def elements_of(type_: "Type") -> int:
    """How many elements one boundary value holds."""
    if not isinstance(type_, TensorType):
        raise ValueError(f"{type_!r} is not one tensor boundary")
    counted = 1
    for extent in type_.shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            raise ValueError(f"{type_!r} has no concrete element count")
        counted *= extent
    return counted


def access_elements(
    relations: AccessRelations, *, boundary: int, output: bool = False
) -> AccessQuantity | None:
    """What the Op said this boundary moves.

    Read back, never re-derived: the handler is the only thing that knows what
    its Op does, and a second opinion computed from the pattern would be a
    guess wearing the same type.
    """
    chosen = relations.outputs if output else relations.inputs
    return chosen[boundary].quantity if 0 <= boundary < len(chosen) else None


@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    One relation per boundary value, in boundary order.

    - ``inputs``: one `BoundaryAccess` per input arg, in argument order.
    - ``outputs``: one per output. Single-output ops have len 1; tuple-output
      ops have one entry per tuple field.
    - ``storage_effect``: where the result's bytes live. An Op that says nothing
      produces its own, because reading an operand at the same indices is not
      the same as being that operand.
    """

    inputs: tuple[BoundaryAccess, ...]
    outputs: tuple[BoundaryAccess, ...]
    storage_effect: "StorageEffectClaim | None" = None

    def __post_init__(self) -> None:
        """Refuse an impossible description here, rather than interpret it later.

        A boundary that says it writes on the input side is a broken handler,
        and a consumer reading it as a read hides the break behind a plausible
        number. What needs the Call -- element width, boundary count -- belongs
        to the registration wrapper, which has one.
        """
        for side in ("inputs", "outputs"):
            stated = getattr(self, side)
            if not isinstance(stated, tuple) or not all(
                isinstance(item, BoundaryAccess) for item in stated
            ):
                raise ValueError(f"{side} is one BoundaryAccess per boundary, got {stated!r}")
        if not self.outputs:
            raise ValueError("an operation produces at least one value to describe")
        for index, boundary in enumerate(self.inputs):
            if boundary.mode not in (AccessMode.READ, AccessMode.TRANSFER):
                raise ValueError(
                    f"input {index} says it does {boundary.mode.value}; an operand "
                    "is read or transferred, never written"
                )
            if boundary.storage is not None:
                raise ValueError(
                    f"input {index} states where bytes live; that is an output's "
                    "answer about its own"
                )
        for index, boundary in enumerate(self.outputs):
            if boundary.mode not in (AccessMode.WRITE, AccessMode.TRANSFER):
                raise ValueError(
                    f"output {index} says it does {boundary.mode.value}; a result "
                    "is written or transferred, never read"
                )
            links = boundary.storage.links if boundary.storage is not None else ()
            if boundary.mode is AccessMode.TRANSFER and not links:
                raise ValueError(
                    f"output {index} transfers but names no source; a transfer "
                    "states which bytes it moves or it states nothing"
                )
            for link in links:
                if link.input >= len(self.inputs):
                    raise ValueError(
                        f"output {index} links to operand {link.input}, and this "
                        f"call has {len(self.inputs)}"
                    )
                if not _shares_domain(link.source, link.output):
                    raise ValueError(
                        f"output {index} links two patterns over different "
                        "domains, so no coordinate of one names a coordinate "
                        "of the other"
                    )


def _affine_domain(pattern: "AccessPattern") -> "isl.set | None":
    """The set a carrier is indexed by, when it can state one."""
    if isinstance(pattern, isl.multi_aff):
        return isl.map.from_multi_aff(pattern).domain()
    if isinstance(pattern, isl.map):
        return pattern.domain()
    return None


def _shares_domain(source: "AccessPattern", output: "AccessPattern") -> bool:
    """Whether two link patterns are indexed by one iteration domain.

    Two affine carriers are compared as the sets they are, not as two rank
    numbers: a pair of rank-2 maps over different extents index different
    programs. Two windows are compared by the extents that are their domain. A
    lookup states no domain of its own, and is admitted -- a link through one is
    refused later, by a prover that has no bijection to offer, rather than here
    where refusing it would make that ruling unwritable.
    """
    if isinstance(source, IndexedAccess) or isinstance(output, IndexedAccess):
        return True
    if isinstance(source, WindowAccess) and isinstance(output, WindowAccess):
        return source.extents == output.extents
    left, right = _affine_domain(source), _affine_domain(output)
    if left is None or right is None:
        return False
    return left.is_equal(right)







@dataclass(frozen=True)
class AccessRelationResult:
    """Forward access relation for one Call, built from input types alone.

    ``domain`` is the bounded iteration domain as an ``isl.set``: static dims
    are constant constraints, dynamic dims are isl parameters. ``maps`` holds
    one access ``isl.map`` per boundary value, in boundary order (inputs
    first, then outputs). ``param_map`` resolves each of ``domain``'s isl
    parameter names back to the ``ShapeDim`` it stands for; it is this
    Call's own data, never shared with any other Call's relation. The
    carrier holds no tensor shape — output shape is typeinfer-side data.
    """

    domain: "isl.set"
    maps: tuple["isl.map", ...]
    param_map: dict = field(default_factory=dict)







class StorageEffectKind(enum.Enum):
    """What one Op claims about where its result's bytes live."""

    PRODUCE = "produce"
    FORWARD = "forward"
    UPDATE = "update"


@dataclass(frozen=True)
class StorageSpan:
    """One run of bytes inside one operand's own buffer."""

    operand: int
    offset: int
    size: int


@dataclass(frozen=True)
class StorageEffectClaim:
    """Which operands a result lives in, and where inside them if that is known.

    ``spans_required`` marks a claim that is only true because of where the
    pieces sit -- putting several of them side by side, say. Without it, a claim
    whose spans do not resolve still stands on its operands alone, which is what
    a re-indexing of one operand is whether or not its address is known.
    """

    kind: StorageEffectKind = StorageEffectKind.PRODUCE
    operands: tuple[int, ...] = ()
    spans: tuple[StorageSpan, ...] = ()
    spans_required: bool = False


access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")




type_relation_registry: AnalysisRegistry = AnalysisRegistry("type_relation")


def _boundaries_of(call, ctx) -> tuple[int, int]:
    """How many boundaries this Call has on each side.

    A tuple result is as many boundaries as it has fields, because each field is
    somewhere of its own that a reader can ask about.
    """
    result = ctx.type_of(call)
    outputs = len(result.fields) if isinstance(result, TupleType) else 1
    return len(call.args), outputs


def _element_bits(type_: "Type") -> int | None:
    """How wide one element of this boundary is, or ``None`` when unknown.

    Bits rather than bytes, because a bool is one bit and a packed float four:
    reading those as "no whole number of bytes, so never mind" let a bool share
    a coordinate with an f32, where one side's index lands 31 bits inside the
    other's element.
    """
    if not isinstance(type_, TensorType):
        return None
    bits = getattr(type_.dtype, "bit_width", None)
    return bits if isinstance(bits, int) and bits > 0 else None


def _field_of(type_: "Type", index: int) -> "Type | None":
    """One field of a tuple, or the value itself when it has no fields."""
    if isinstance(type_, TupleType):
        return type_.fields[index] if 0 <= index < len(type_.fields) else None
    return type_ if index == 0 else None


def _check_links_against(call, ctx, relations: AccessRelations, op_cls: type) -> None:
    """Hold every storage link to the Call whose operands it names."""
    result = ctx.type_of(call)
    for index, boundary in enumerate(relations.outputs):
        links = boundary.storage.links if boundary.storage is not None else ()
        for link in links:
            operand = ctx.type_of(call.args[link.input])
            if isinstance(operand, TupleType):
                if link.input_field is None or not (
                    0 <= link.input_field < len(operand.fields)
                ):
                    raise ValueError(
                        f"{op_cls.__name__} links output {index} to operand "
                        f"{link.input}, which holds {len(operand.fields)} fields, "
                        f"and names field {link.input_field!r}"
                    )
            elif link.input_field is not None:
                raise ValueError(
                    f"{op_cls.__name__} links output {index} to field "
                    f"{link.input_field} of operand {link.input}, which has no "
                    "fields of its own"
                )
            source = _element_bits(_field_of(operand, link.input_field or 0))
            destination = _element_bits(_field_of(result, index))
            if source is None or destination is None:
                raise ValueError(
                    f"{op_cls.__name__} shares output {index} with operand "
                    f"{link.input}, and one of them states no element width; "
                    "bytes cannot be shared between values of unknown shape"
                )
            if source != destination:
                raise ValueError(
                    f"{op_cls.__name__} shares output {index} with operand "
                    f"{link.input}, whose elements are {source} bits against "
                    f"{destination}; one side's coordinate would land inside "
                    "the other's element"
                )


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a GLOBAL-level access-relation handler.

    The handler signature is ``(call, ctx) -> AccessRelations``. Every boundary
    gets a pattern: isl where the addresses are affine, ``IndexedAccess`` or
    ``WindowAccess`` where a runtime value decides them.

    What comes back is held to the Call it was asked about: one entry per
    operand and one per output, and every link's element width against the
    operand it names. Anything needing the Call is checked here, because the
    record itself has no Call to ask.
    """

    def decorate(handler: Callable) -> Callable:
        def checked(call, ctx) -> AccessRelations:
            relations = handler(call, ctx)
            wanted_inputs, wanted_outputs = _boundaries_of(call, ctx)
            for side, stated, wanted in (
                ("input", len(relations.inputs), wanted_inputs),
                ("output", len(relations.outputs), wanted_outputs),
            ):
                if stated != wanted:
                    raise ValueError(
                        f"{op_cls.__name__} describes {stated} {side} "
                        f"boundar{'y' if stated == 1 else 'ies'} of a call with "
                        f"{wanted}"
                    )
            _check_links_against(call, ctx, relations, op_cls)
            return relations

        checked.__name__ = getattr(handler, "__name__", "checked")
        checked.__doc__ = handler.__doc__
        access_relation_registry.register(op_cls, checked)
        return handler

    return decorate


def storage_effect_of(relations: AccessRelations) -> "StorageEffectClaim | None":
    """The legacy whole-Call claim, read off the per-boundary links.

    One truth, two shapes. The claim predates links and still has consumers, so
    it is derived here rather than written a second time by each handler: two
    hand-maintained statements of one fact drift, and the drift is invisible
    until a number is wrong. Spans stay with the handler that computes byte
    offsets until the allocation does, which is the step that retires this.

    A claim covers the whole result, so an output that shares only part of
    itself, or a tuple whose fields disagree, has no single claim to make.
    """
    if len(relations.outputs) != 1:
        return None
    boundary = relations.outputs[0]
    links = boundary.storage.links if boundary.storage is not None else ()
    if not links:
        return None
    kinds = {link.kind for link in links}
    if len(kinds) != 1:
        return None
    kind = (
        StorageEffectKind.UPDATE if kinds == {"preserve"} else StorageEffectKind.FORWARD
    )
    return StorageEffectClaim(kind, tuple(link.input for link in links))


def declared_storage(call, ctx) -> "StorageEffectClaim | None":
    """What one Call's Op claims about where its result's bytes live.

    Asking the Op means building its relations, because one registration answers
    both questions. A handler that cannot build them says so and is heard: the
    fail-closed direction belongs to the proof, not to the claim, or a broken
    handler would read as an Op that merely produces.
    """
    handler = access_relation_registry.lookup(type(call.target))
    return None if handler is None else handler(call, ctx).storage_effect


def elementwise_elements(arg, call, ctx) -> int:
    """What one participant reads of *arg* to produce its share of *call*.

    A participant reads no more of an operand than it produces, and no more than
    the operand holds -- the second half being what makes a broadcast operand
    cost its own smaller size. The bound matters most where an operand is not
    sharded at all: a replicated source projects to the whole of itself, and
    charging that to every participant multiplies one read by the number of them.
    """
    type_ = ctx.local_type_of(arg)
    if not isinstance(type_, TensorType):
        return 0
    produced = ctx.local_type_of(call)
    if not isinstance(produced, TensorType):
        return elements_of(type_)
    return min(elements_of(type_), elements_of(produced))


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


def measures_without_reading(call, ctx) -> AccessRelations:
    """An Op that answers from a value's Type rather than from its elements.

    A rank, a shape, a name for the same value at another level: the answer is
    already in the Type, so no coordinate is read. The relation says that -- an
    empty map, nothing crossing -- rather than an identity claiming a read the
    Op never performs.
    """
    result = ctx.local_type_of(call)
    out_rank = len(result.shape) if hasattr(result, "shape") else 0

    def _empty(arg) -> "isl.map":
        type_ = ctx.local_type_of(arg)
        in_rank = len(type_.shape) if hasattr(type_, "shape") else 0
        reads = ", ".join(f"i{index}" for index in range(in_rank))
        domain = ", ".join(f"d{index}" for index in range(out_rank))
        return isl.map(f"{{ [{domain}] -> [{reads}] : 1 = 0 }}")

    return AccessRelations(
        inputs=tuple(moves(_empty(arg), 0) for arg in call.args),
        outputs=(writes(_empty(call.args[0]) if call.args else _identity(out_rank), 0),),
    )


def linearized_view(out_shape: tuple, in_shape: tuple) -> "isl.multi_aff":
    """Where an output coordinate sits in a source of another shape.

    A reshape keeps the elements in the order they were in and renames the axes
    over them, so one flat index answers both sides. The domain is the output's,
    because that is the side a reader walks. An empty shape holds nothing, so
    nothing of it is anywhere in the source: the answer is an empty relation
    rather than a division by an axis of length zero.
    """
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0
        for extent in (*out_shape, *in_shape)
    ):
        raise ValueError(
            f"a view relabels a shape it can count: {out_shape!r} from {in_shape!r}"
        )
    out_rank, in_rank = len(out_shape), len(in_shape)
    if 0 in out_shape or 0 in in_shape:
        dims = ", ".join(f"d{index}" for index in range(out_rank))
        reads = ", ".join("0" for _ in range(in_rank))
        return isl.map(f"{{ [{dims}] -> [{reads}] : 1 = 0 }}")
    dims = [f"d{index}" for index in range(out_rank)]
    flat, stride = [], 1
    for index in reversed(range(out_rank)):
        flat.append(f"{dims[index]}" if stride == 1 else f"{stride} * {dims[index]}")
        stride *= out_shape[index]
    linear = " + ".join(reversed(flat)) if flat else "0"
    reads, stride = [], 1
    strides = []
    for extent in reversed(in_shape):
        strides.append(stride)
        stride *= extent
    for axis, step in zip(range(in_rank), reversed(strides)):
        term = f"({linear})" if step == 1 else f"floor(({linear}) / {step})"
        reads.append(term if in_shape[axis] == stride // step else f"({term}) mod {in_shape[axis]}")
    domain = ", ".join(dims)
    return isl.multi_aff(f"{{ [{domain}] -> [{', '.join(reads)}] }}")


def view_relations(
    source: int = 0,
    storage: "Callable[..., StorageEffectClaim | None] | None" = None,
    mapping: "Callable[..., tuple[AccessPattern, AccessPattern]] | None" = None,
    field: "Callable[..., int | None] | None" = None,
) -> Callable[..., AccessRelations]:
    """An Op whose whole purpose is to re-address one operand's bytes.

    A reshape, a slice, a reshard, an item of a tuple: the result is those same
    elements under another name. Both boundaries are transfers, and the output
    names the operand it came from through one forward link, so a plan that can
    put them at the same addresses makes this cost nothing and a plan that
    cannot makes it a copy. Neither answer is this handler's to give.
    """

    def _handler(call, ctx) -> AccessRelations:
        result = ctx.local_type_of(call)
        out_rank = len(result.shape) if hasattr(result, "shape") else 0
        moved = elements_of(result)
        held = AccessQuantity(moved, moved)

        def _rank_of(arg) -> int:
            type_ = ctx.local_type_of(arg)
            return len(type_.shape) if hasattr(type_, "shape") else out_rank

        reads, written = (
            mapping(call, ctx)
            if mapping is not None
            else (_identity(out_rank), _identity(out_rank))
        )
        link = StorageLink(
            kind="forward",
            input=source,
            source=reads,
            output=written,
            quantity=held,
            input_field=None if field is None else field(call, ctx),
        )
        return AccessRelations(
            inputs=tuple(
                BoundaryAccess(
                    reads if index == source else _identity(_rank_of(arg)),
                    held if index == source else AccessQuantity(0, 0),
                    AccessMode.TRANSFER if index == source else AccessMode.READ,
                )
                for index, arg in enumerate(call.args)
            ),
            outputs=(transfers(written, held, link),),
            storage_effect=None if storage is None else storage(call, ctx),
        )

    return _handler


def identity_relations(
    n_inputs: int, storage: "Callable[..., StorageEffectClaim | None] | None" = None
) -> Callable[..., AccessRelations]:
    """Identity relations.

    Factory for a GLOBAL-level access-relation handler whose ``n_inputs``
    inputs and single output are all elementwise identity.

    Each input contributes its own-rank identity; the output uses its own
    rank. A structural (non-tensor) input arg — e.g. ``TupleGetItem``'s tuple
    operand — has no shape of its own, so it borrows the output's rank.
    """

    def _handler(call, ctx) -> AccessRelations:
        result = ctx.local_type_of(call)
        out_rank = len(result.shape)

        def _rank_of(arg) -> int:
            ty = ctx.local_type_of(arg)
            return len(ty.shape) if hasattr(ty, "shape") else out_rank

        return AccessRelations(
            inputs=tuple(
                moves(
                    _identity(_rank_of(call.args[index])),
                    elementwise_elements(call.args[index], call, ctx),
                )
                for index in range(n_inputs)
            ),
            outputs=(writes(_identity(out_rank), elements_of(result)),),
            storage_effect=None if storage is None else storage(call, ctx),
        )

    return _handler


def static_bytes(type_: "Type") -> int | None:
    """How many bytes a Type holds, or ``None`` when it is not static."""
    if isinstance(type_, TupleType):
        sizes = [static_bytes(field_) for field_ in type_.fields]
        if any(size is None for size in sizes):
            return None
        return sum(size for size in sizes if size is not None)
    if not isinstance(type_, TensorType):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in type_.shape):
        return None
    try:
        amount = tensor_bytes(type_)
    except (TypeError, ValueError):
        return None
    return amount if isinstance(amount, int) else None


def dense(type_: "Type") -> bool:
    """Whether a Type's own elements sit in one unbroken row-major run.

    A sharded Type is read through its per-position tile, which is the run one
    position addresses. Anything this cannot decide -- a composed layout, a
    symbolic stride -- is not dense here, so the proof that needed it fails.
    """
    if not isinstance(type_, TensorType):
        return False
    layout = type_.layout
    shard = shard_layout_of(layout)
    if shard is not None:
        layout = shard.layout
    if layout is None:
        return True
    if not isinstance(layout, Layout):
        return False
    if layout.strides is None:
        return True
    expected = try_c_order_strides(flatten(layout.shape))
    return expected is not None and tuple(layout.strides) == expected


def same_placement(left: "Type", right: "Type") -> bool:
    """Whether two Types name the same storage, element size, and positions."""
    if not (isinstance(left, TensorType) and isinstance(right, TensorType)):
        return False
    if left.storage != right.storage or left.dtype != right.dtype:
        return False
    left_shard, right_shard = shard_layout_of(left.layout), shard_layout_of(right.layout)
    if left_shard is None or right_shard is None:
        return left_shard is None and right_shard is None
    return left_shard.mesh == right_shard.mesh


def forward_whole(call, operand: int, ctx) -> StorageEffectClaim:
    """Forward all of *operand*, with its address when the sizes are static."""
    size = static_bytes(ctx.type_of(call))
    if size is None or size != static_bytes(ctx.type_of(call.args[operand])):
        return StorageEffectClaim(StorageEffectKind.FORWARD, (operand,))
    return StorageEffectClaim(
        StorageEffectKind.FORWARD, (operand,), (StorageSpan(operand, 0, size),)
    )


def update_destination(call, ctx, *, destination: int) -> "StorageEffectClaim | None":
    """Claim that the result is the destination's buffer after an overwrite.

    Whether overwriting is free is not a question about this Call: it depends on
    who else still reads those bytes and on whose buffer it is. All this states
    is that the result is that operand, laid out the same way and the same size.
    """
    if not same_placement(ctx.type_of(call.args[destination]), ctx.type_of(call)):
        return None
    size = static_bytes(ctx.type_of(call))
    if size is None or size != static_bytes(ctx.type_of(call.args[destination])):
        return None
    return StorageEffectClaim(
        StorageEffectKind.UPDATE, (destination,), (StorageSpan(destination, 0, size),)
    )


def register_type_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register a forward type-relation builder.

    The handler signature is ``(call, input_types, ctx) -> AccessRelationResult``.
    It reads only ``input_types`` and the op's attributes — never the Call's own
    output type — so it can run before the output type exists.
    """
    return type_relation_registry.decorator()(op_cls)


def build_relation(call, input_types, ctx) -> "AccessRelationResult | None":
    """Build relation.

    Build the forward access relation for *call*, or ``None`` if its op has
    no registered builder.
    """
    fn = type_relation_registry.lookup(type(call.target))
    if fn is None:
        return None
    return fn(call, input_types, ctx)


__all__ = [
    "StorageEffectClaim",
    "StorageEffectKind",
    "StorageSpan",
    "declared_storage",
    "dense",
    "forward_whole",
    "same_placement",
    "static_bytes",
    "update_destination",
    "AccessPattern",
    "AccessQuantity",
    "BoundaryAccess",
    "IndexedAccess",
    "elements_of",
    "moves",
    "moves_between",
    "OperandValue",
    "access_elements",
    "WindowAccess",
    "AccessRelations",
    "AccessRelationResult",
    "access_relation_registry",
    "AccessMode",
    "OutputStorage",
    "StorageLink",
    "elementwise_elements",
    "storage_effect_of",
    "linearized_view",
    "view_relations",
    "transfers",
    "writes",
    "measures_without_reading",
    "type_relation_registry",
    "register_access_relation",
    "register_type_relation",
    "identity_relations",
    "build_relation",
]
