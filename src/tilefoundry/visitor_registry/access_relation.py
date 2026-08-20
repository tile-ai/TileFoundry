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

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Callable, Union

import isl

from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.shard_layout import layout_axis_to_tensor_axis, shard_layout_of

from .registries import AnalysisRegistry


@dataclass(frozen=True)
class IndexedAccess:
    """One boundary reached through the values of another operand.

    A gather names the coordinate it wants by an element of ``index_operand``,
    so no address is known here. What is being indexed is the boundary this
    pattern sits on -- boundaries are already in order and already say which
    value they describe -- so a target field would be a second name for
    something already said, and a sentinel for "this output" would be a third.
    """

    index_operand: int
    axis: int

    def __post_init__(self) -> None:
        for name in ("index_operand", "axis"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"a lookup names its {name.replace('_', ' ')} by position, "
                    f"not {value!r}"
                )


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
    extents: tuple["OperandValue | ShapeDim", ...]
    complement: bool = False


@dataclass(frozen=True)
class AffineAccess:
    """One boundary's relation, together with what its parameters are.

    A coordinate an Op only learns at run time is a parameter of the relation
    rather than a hole in it: which coordinates are reached depends on the value,
    and the relation says so by naming the value it depends on. Each entry pairs
    the parameter's name in *relation* with the operand element or dimension it
    is, so whoever restricts the relation can bind it rather than guess it.

    A relation with no parameters is the ordinary case and states none.
    """

    relation: "isl.map"
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.relation, isl.map):
            raise ValueError(
                f"an affine access is a relation, not {self.relation!r}"
            )
        named = {
            self.relation.get_dim_name(isl.dim_type.PARAM, index)
            for index in range(self.relation.dim(isl.dim_type.PARAM))
        }
        bound: dict[str, object] = {}
        for name, value in self.parameters:
            if name in bound and bound[name] is not value:
                raise ValueError(
                    f"an affine access binds {name!r} to two different values; one "
                    f"name in one relation is one value"
                )
            bound[name] = value
        if named != set(bound):
            raise ValueError(
                f"an affine access binds {sorted(bound)} but its relation names "
                f"{sorted(named)}; a parameter nobody can bind is a hole"
            )


AccessPattern = Union[
    "isl.multi_aff", "isl.map", AffineAccess, IndexedAccess, WindowAccess
]







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

    ``where`` reads an output coordinate and answers with the input coordinate
    holding it: a pattern rather than a byte offset, which could state neither
    an unbound window start nor the mapping a transpose forwards through, and
    one rather than two, which would be the same answer twice. ``quantity`` is
    how much the region holds, never measured off a pattern. A link is a
    candidate: whether these bytes really are shared is the allocation's answer,
    and an unhonoured link is the copy instead.
    """

    kind: str
    input: int
    where: "AccessPattern"
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
        if not isinstance(
            self.where,
            (isl.multi_aff, isl.map, AffineAccess, IndexedAccess, WindowAccess),
        ):
            raise ValueError(
                f"a link reads through a relation, a lookup or a window, not "
                f"through {self.where!r}"
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
            self.pattern,
            (isl.multi_aff, isl.map, AffineAccess, IndexedAccess, WindowAccess),
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


def _check_lookups_against(call, ctx, relations: AccessRelations, op_cls: type) -> None:
    """Hold every lookup to the operand it indexes through and the axis it names.

    A lookup says nothing checkable on its own -- two non-negative numbers -- so
    the record cannot refuse an index operand that is not there or an axis the
    boundary's own Type does not have. Here both are known.
    """
    result = ctx.type_of(call)
    sides = (
        ("input", relations.inputs, [ctx.type_of(arg) for arg in call.args]),
        (
            "output",
            relations.outputs,
            list(result.fields) if isinstance(result, TupleType) else [result],
        ),
    )
    def hold(lookup: IndexedAccess, where: str, held) -> None:
        if lookup.index_operand >= len(call.args):
            raise ValueError(
                f"{op_cls.__name__} indexes {where} through operand "
                f"{lookup.index_operand}, and this call has {len(call.args)}"
            )
        rank = len(held.shape) if isinstance(held, TensorType) else 0
        if lookup.axis >= rank:
            raise ValueError(
                f"{op_cls.__name__} indexes {where} along axis {lookup.axis}, "
                f"and it has {rank}"
            )

    for side, boundaries, types in sides:
        for index, boundary in enumerate(boundaries):
            held = types[index] if index < len(types) else None
            if isinstance(boundary.pattern, IndexedAccess):
                hold(boundary.pattern, f"{side} {index}", held)
            links = boundary.storage.links if boundary.storage is not None else ()
            for link in links:
                for end, pattern in (("where", link.where),):
                    if isinstance(pattern, IndexedAccess):
                        reached = _field_of(
                            ctx.type_of(call.args[link.input]),
                            link.input_field or 0,
                        )
                        hold(pattern, f"{side} {index}'s link {end}", reached)


def _image_rank(pattern: "AccessPattern") -> int | None:
    """How many coordinates a carrier names, or ``None`` when it names none.

    A window names one offset and one extent per coordinate, so it answers the
    same question an affine image does and is held to the same rank. Both being
    checked is what keeps a boundary from carrying a logical window against a
    factored image of the same value.
    """
    if isinstance(pattern, AffineAccess):
        return pattern.relation.dim(isl.dim_type.OUT)
    if isinstance(pattern, (isl.multi_aff, isl.map)):
        return pattern.dim(isl.dim_type.OUT)
    if isinstance(pattern, WindowAccess):
        if len(pattern.offsets) != len(pattern.extents):
            raise ValueError(
                f"a window states {len(pattern.offsets)} offsets and "
                f"{len(pattern.extents)} extents"
            )
        return len(pattern.extents)
    return None


def _check_image_ranks(call, ctx, relations: AccessRelations, op_cls: type) -> None:
    """Every affine image names one coordinate per position of what it reaches.

    An image is composed with a `Layout`, whose shape and strides are the value's
    factored positions, so another rank cannot become an address at all. The rank
    to match is this view's. A boundary that moves nothing is exempt: an Op
    answering from a Type without reading it states an empty map of whatever
    shape reads clearest.
    """
    result = ctx.local_type_of(call)
    sides = (
        ("input", relations.inputs, [ctx.local_type_of(arg) for arg in call.args]),
        (
            "output",
            relations.outputs,
            list(result.fields) if isinstance(result, TupleType) else [result],
        ),
    )
    for side, boundaries, types in sides:
        for index, boundary in enumerate(boundaries):
            held = types[index] if index < len(types) else None
            if isinstance(held, TupleType):
                continue
            wanted = len(held.shape) if isinstance(held, TensorType) else 0
            if boundary.quantity.upper:
                _hold_rank(
                    boundary.pattern, wanted, f"{side} {index}", op_cls
                )
            links = boundary.storage.links if boundary.storage is not None else ()
            for link in links:
                if not link.quantity.upper:
                    continue
                source = ctx.local_type_of(call.args[link.input])
                if link.input_field is not None and isinstance(source, TupleType):
                    source = source.fields[link.input_field]
                _hold_rank(
                    link.where,
                    len(source.shape) if isinstance(source, TensorType) else 0,
                    f"{side} {index}'s link",
                    op_cls,
                )
                _hold_domain(link.where, wanted, f"{side} {index}'s link", op_cls)



def _hold_rank(pattern: "AccessPattern", wanted: int, where: str, op_cls: type) -> None:
    """Refuse an affine image that names a different number of coordinates."""
    stated = _image_rank(pattern)
    if stated is not None and stated != wanted:
        raise ValueError(
            f"{op_cls.__name__} reads {where} at {stated} coordinates, and it "
            f"has {wanted} in this view"
        )


def _domain_rank(pattern: "AccessPattern") -> int | None:
    """How many coordinates a carrier is asked by, or None when it says none.

    A window is asked by one offset and one extent per coordinate of the value
    it is a window of, which is the same question an affine domain answers.
    """
    if isinstance(pattern, AffineAccess):
        return pattern.relation.dim(isl.dim_type.IN)
    if isinstance(pattern, (isl.multi_aff, isl.map)):
        return pattern.dim(isl.dim_type.IN)
    if isinstance(pattern, WindowAccess):
        return _image_rank(pattern)
    return None


def _hold_domain(pattern: "AccessPattern", wanted: int, where: str, op_cls: type) -> None:
    """Refuse a link asked by a different number of coordinates than it has.

    A link answers for one coordinate of its output, so it is asked by as many
    as that output has. One asked by fewer is a link to some other value, and
    the reader that composes it with this occurrence finds that out too late to
    say whose bytes it was talking about.
    """
    stated = _domain_rank(pattern)
    if stated is not None and stated != wanted:
        raise ValueError(
            f"{op_cls.__name__} links {where} from {stated} coordinates, and it "
            f"has {wanted} in this view"
        )


def _check_claim_against_links(relations: AccessRelations, op_cls: type) -> None:
    """Hold the legacy whole-Call claim to the links of the same output.

    One fact in two shapes, and the older one still has consumers. A claim that
    no link supports is drift, and drift is invisible until a number is wrong.
    The converse is allowed and is not drift: a link is a candidate and a claim
    is a conclusion, so a reshard across levels states its link and no claim.
    """
    stated = relations.storage_effect
    if stated is None:
        return
    derived = storage_effect_of(relations)
    if derived is None:
        raise ValueError(
            f"{op_cls.__name__} claims {stated.kind.value} storage in operands "
            f"{stated.operands}, and no link of its output says so"
        )
    if (stated.kind, stated.operands) != (derived.kind, derived.operands):
        raise ValueError(
            f"{op_cls.__name__} claims {stated.kind.value} storage in operands "
            f"{stated.operands}, while its links say {derived.kind.value} in "
            f"{derived.operands}"
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
            _check_lookups_against(call, ctx, relations, op_cls)
            _check_image_ranks(call, ctx, relations, op_cls)
            _check_claim_against_links(relations, op_cls)
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


def logical_axes_of(local: "Type", logical: "Type") -> list[int]:
    """Which logical axis each axis of a projected Type belongs to.

    A canonical `ShardLayout` factors a logical axis into several layout
    positions -- an extent of 12 over a mesh of 6 becomes `(6, 2)` -- and the
    projected Type keeps those positions. So the projected rank is not the
    authored rank and an Op's own axis numbers do not index it: reducing "axis
    1" by position would reduce the mesh factor of axis 0 and carry its residual
    through. Amounts can stay right while that happens; a mapping cannot.
    """
    layout = getattr(local, "layout", None)
    inner = getattr(layout, "layout", None)
    shape = getattr(inner, "shape", None) or getattr(layout, "shape", None)
    if shape is None or len(shape) != len(local.shape):
        return list(range(len(local.shape)))
    return layout_axis_to_tensor_axis(tuple(shape), tuple(logical.shape))


def logical_coordinates(local: "Type", logical: "Type") -> dict[int, str]:
    """One expression per logical axis, rebuilt from the positions holding it.

    The inverse of `factored_image`: a domain is indexed by a projected Type's
    positions, and an Op reasons about the logical axes those positions came
    from. A position a participant holds one of contributes nothing, its
    coordinate being fixed; the rest are recombined in the order the layout
    states them.
    """
    belongs = logical_axes_of(local, logical)
    linear: dict[int, str] = {}
    strides: dict[int, int] = {}
    for position in reversed(range(len(belongs))):
        owner = belongs[position]
        extent = local.shape[position]
        if extent == 1:
            continue
        stride = strides.get(owner, 1)
        term = f"d{position}" if stride == 1 else f"{stride} * d{position}"
        linear[owner] = term if owner not in linear else f"{linear[owner]} + {term}"
        strides[owner] = stride * extent
    return linear


def holds_whole_axis(local: "Type", logical: "Type", axis: int) -> bool:
    """Whether a participant holds all of one logical axis.

    An Op that narrows or divides an axis states its offsets against the whole
    of it. A participant holding a slice of that axis would need to know which
    slice -- its own offset -- to say where those offsets land, and a projection
    has extents and no offsets. So the question is asked, and the Op refuses
    what it cannot say rather than answering for the first participant.
    """
    held = 1
    for position, owner in enumerate(logical_axes_of(local, logical)):
        if owner == axis:
            held *= local.shape[position]
    return held == logical.shape[axis]


def affine_term(value, name: str) -> "tuple[str, tuple[tuple[str, object], ...]]":
    """One number of a relation, as a coefficient or as a bound parameter.

    A number written down is a coefficient of the map. Anything else is a
    parameter carrying the value it is, so whoever restricts the relation
    resolves it rather than reading its spelling. The caller states what it
    guarantees about the parameter; nothing is guaranteed here.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value), ()
    return name, ((name, value),)


def _at_most(extent: str, limits: tuple, position: int) -> list[str]:
    """The most a window may extend on one axis, when the Op guarantees one.

    A runtime extent with no stated ceiling is a window a reader can only union
    over the whole axis. The Op's own contract usually says more than that, and
    saying it here is what keeps one relation answering both how much this
    occurrence moves and how much the axis it walks ever holds.
    """
    limit = limits[position] if position < len(limits) else None
    return [] if limit is None else [f"{extent} <= {limit}"]


def placed_window(
    offsets: tuple, extents: tuple, rank: int, limits: tuple = (), within: tuple = ()
) -> tuple:
    """What a window reaches among a value's own positions, and what it leaves.

    One domain for both, the value's own positions, so the two answer about the
    same coordinates: the window is where the offsets put it, and what is left
    alone is every position it does not cover -- a difference, not a flag. An
    offset or an extent only known later is a parameter bound to the value it is,
    kept inside whatever the Op guarantees about it, and a window begins and ends
    inside what it is placed in -- so one covering the whole of something leaves
    none of it, not whatever an offset nobody could pass would have left.
    """
    domain = ", ".join(f"d{index}" for index in range(rank))
    guards: list[str] = []
    parameters: list[tuple[str, object]] = []
    for position in range(rank):
        begin, bound_begin = affine_term(offsets[position], f"o{position}")
        extent, bound_extent = affine_term(extents[position], f"e{position}")
        parameters.extend((*bound_begin, *bound_extent))
        if bound_begin:
            guards.append(f"0 <= {begin}")
        if bound_extent:
            guards.append(f"1 <= {extent}")
            guards.extend(_at_most(extent, limits, position))
        if (bound_begin or bound_extent) and position < len(within):
            whole, bound_whole = affine_term(within[position], f"w{position}")
            parameters.extend(bound_whole)
            guards.append(f"{begin} + {extent} <= {whole}")
        if begin == "0":
            guards.append(f"0 <= d{position} < {extent}")
            continue
        guards.append(f"{begin} <= d{position} < {begin} + {extent}")
    prefix = isl_parameters(parameters)
    where = f" : {' and '.join(guards)}" if guards else ""
    reached = isl.map(f"{prefix}{{ [{domain}] -> [{domain}]{where} }}")
    whole = isl.map(f"{prefix}{{ [{domain}] -> [{domain}] }}")
    left = whole.subtract(reached).intersect_params(reached.params())
    return (
        AffineAccess(left, tuple(parameters)),
        AffineAccess(reached, tuple(parameters)),
    )


def iterating(extents: "Sequence", relations: "AccessRelations") -> "AccessRelations":
    """Every boundary of one Op, on the iteration space that Op walks.

    An access map's domain is the Op's whole iteration space, so its bounds are
    the Op's and every boundary shares them -- not each boundary's own, and not
    a reader's guess from a result's rank. A contraction walks the axis it
    contracts; most Ops walk what they produce. A boundary may be partial in
    that space, which is one relation empty somewhere, not a second space.
    """
    domain = index_set(tuple(extents))
    if domain is None:
        return relations
    return AccessRelations(
        inputs=tuple(_held_to(boundary, domain) for boundary in relations.inputs),
        outputs=tuple(_held_to(boundary, domain) for boundary in relations.outputs),
        storage_effect=relations.storage_effect,
    )


def _held_to(boundary: "BoundaryAccess", domain: "isl.set") -> "BoundaryAccess":
    """One boundary, restricted to the coordinates its Op iterates."""
    pattern = boundary.pattern
    if not isinstance(pattern, (AffineAccess, isl.map, isl.multi_aff)):
        return boundary
    relation = relation_of(pattern)
    if relation.dim(isl.dim_type.IN) != domain.dim(isl.dim_type.SET):
        raise ValueError(
            f"a boundary is asked by {relation.dim(isl.dim_type.IN)} coordinates "
            f"and its Op iterates {domain.dim(isl.dim_type.SET)}; one Op states "
            "one coordinate system and every boundary of it answers about that one"
        )
    held = relation.intersect_domain(domain)
    named = {
        held.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(held.dim(isl.dim_type.PARAM))
    }
    stated = pattern.parameters if isinstance(pattern, AffineAccess) else ()
    return dataclasses.replace(
        boundary,
        pattern=AffineAccess(
            held, tuple((name, value) for name, value in stated if name in named)
        ),
    )


def relation_of(pattern: "AccessPattern") -> "isl.map":
    """One boundary's coordinates as a relation, whatever carrier states them.

    A function and a relation answer the same question about the same two spaces,
    so a reader that only asks "where does this coordinate go" should not have to
    know which one it was handed.
    """
    if isinstance(pattern, AffineAccess):
        pattern = pattern.relation
    return isl.map.from_multi_aff(pattern) if isinstance(pattern, isl.multi_aff) else pattern


def index_set(shape) -> "isl.set | None":
    """The coordinates one value legally has, or nothing when its shape is not numbers."""
    if any(
        not isinstance(extent, int) or isinstance(extent, bool) or extent < 0
        for extent in shape
    ):
        return None
    if not shape:
        return isl.set("{ [] }")
    names = ", ".join(f"d{index}" for index in range(len(shape)))
    guards = " and ".join(f"0 <= d{index} < {extent}" for index, extent in enumerate(shape))
    return isl.set(f"{{ [{names}] : {guards} }}")


def _as_number(value) -> int | None:
    """The number a bound parameter's value is, when it is one."""
    number = static_dim_value(value)
    if number is not None:
        return number
    inner = getattr(value, "value", None)
    return inner if isinstance(inner, int) and not isinstance(inner, bool) else None


def settled(pattern: "AccessPattern") -> "isl.map":
    """One relation with every parameter fixed to a number.

    A parameter bound to something that has a value is fixed to that value. One
    whose value nobody here holds is fixed to the smallest the relation itself
    allows: the first legal iteration of a loop, the smallest legal window of a
    runtime extent. Either way a reader gets a number it can check rather than a
    range it has to interpret, so the parameters leave with their values put in.
    A relation nothing satisfies has no value to settle on, and reaches nothing
    whatever a reader would have picked.
    """
    relation = relation_of(pattern)
    bound = dict(pattern.parameters) if isinstance(pattern, AffineAccess) else {}
    names = [
        relation.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(relation.dim(isl.dim_type.PARAM))
    ]
    if not names:
        return relation
    if relation.is_empty():
        return relation.project_out(isl.dim_type.PARAM, 0, len(names))
    space = f"[{', '.join(names)}] -> "
    legal = relation.params()
    for name in names:
        number = _as_number(bound.get(name))
        if number is None:
            probe = isl.set(f"{space}{{ [x] : x = {name} }}").intersect_params(legal)
            smallest = str(probe.dim_min_val(0))
            try:
                number = int(smallest)
            except ValueError:
                raise ValueError(
                    f"a relation leaves {name!r} at {smallest} because it does not "
                    "state what that parameter may be; a boundary nobody can bind "
                    "is not one a reader can count"
                ) from None
        legal = legal.intersect(isl.set(f"{space}{{ : {name} = {number} }}"))
    settled_at = relation.intersect_params(legal)
    return settled_at.project_out(isl.dim_type.PARAM, 0, len(names))


def reached_elements(
    pattern: "AccessPattern", box: "isl.set | None" = None, within: "isl.set | None" = None
) -> int | None:
    """How many distinct boundary elements one boundary reaches.

    A boundary's quantity is not a second field to keep in step: it is what the
    relation says over the coordinates its Op iterates, which the relation itself
    carries. Reaching the same element from many coordinates is one element
    moved, not many dependences, so an inner iteration axis costs nothing; and
    reaching past the coordinates the operand has is not reaching at all.
    """
    relation = settled(pattern)
    if within is not None:
        relation = relation.intersect_domain(within)
    image = relation.range()
    if box is not None and box.tuple_dim() == image.tuple_dim():
        image = image.intersect(box)
    if image.dim(isl.dim_type.PARAM):
        return None
    try:
        return int(str(image.coalesce().count_val()))
    except ValueError:
        return None


def control_leaves(ctx, arg) -> int:
    """How many numbers one operand carries for placing or sizing a window.

    A window is placed by one number per axis it is placed on, and an operand
    holding several of them carries several: a tuple of offsets is read once per
    field, not once.
    """
    stated = ctx.type_of(arg)
    return len(stated.fields) if isinstance(stated, TupleType) else 1


def _control_space(rank: int, ctx, arg) -> "tuple[str, str, str]":
    """The domain, image and reach of one control operand's own coordinates.

    A tuple of numbers is indexed flat, one leaf per field. A lone scalar's legal
    index set is the single point, at whatever positions its own view gives it.
    """
    domain = ", ".join(f"d{index}" for index in range(rank))
    stated = ctx.type_of(arg)
    if isinstance(stated, TupleType):
        return domain, "l", f"0 <= l < {len(stated.fields)}"
    held = ctx.local_type_of(arg)
    return domain, ", ".join("0" for _ in range(len(getattr(held, "shape", ()) or ()))), ""


def control_read(rank: int, ctx, arg) -> "AffineAccess":
    """The control numbers one operand carries, each read once.

    The domain is the result's positions like every other boundary, because a
    reader applies one execution domain to all of them and a boundary with a rank
    of its own is one it cannot answer. What it reaches is one point per number,
    so a reader counts the numbers rather than believing an empty set.
    """
    domain, image, reach = _control_space(rank, ctx, arg)
    where = f" : {reach}" if reach else ""
    return AffineAccess(isl.map(f"{{ [{domain}] -> [{image}]{where} }}"))


def addresses_only(rank: int, ctx, arg) -> "AffineAccess":
    """An operand a view is addressed with and does not read.

    A slice's bounds say which bytes the result already is; being handed them is
    not reading them. The empty relation over the same coordinates is that
    answer -- not an identity over the result, which would charge a whole window
    to a scalar nobody moved.
    """
    domain, image, _reach = _control_space(rank, ctx, arg)
    return AffineAccess(isl.map(f"{{ [{domain}] -> [{image}] : false }}"))


def reached_at(
    rank: int,
    local: "Type",
    logical: "Type",
    reads: dict,
    free: tuple = (),
) -> "AffineAccess":
    """The coordinates one operand is reached at, stated per logical axis.

    An Op reasons in logical axes and a participant is indexed by the positions
    its layout made, so the expressions are given per axis and spread over the
    positions holding it. An axis named `free` is one whose coordinate is a value
    nobody has here -- the element a lookup read decides it -- so the relation
    covers every coordinate that axis could legally name instead of guessing one.
    That keeps the answer bounded, and countable, without a carrier a reader has
    to special-case.
    """
    belongs = logical_axes_of(local, logical)
    stated = [reads.get(axis, "0") for axis in range(len(logical.shape))]
    for axis in free:
        stated[axis] = "0"
    image = factored_image(stated, local, logical)
    parameters: list[tuple[str, object]] = []
    guards: list[str] = []
    for position, owner in enumerate(belongs):
        if owner not in free or local.shape[position] == 1:
            continue
        extent, bound = affine_term(local.shape[position], f"n{position}")
        parameters.extend(bound)
        image[position] = f"g{position}"
        guards.append(f"0 <= g{position} < {extent}")
    domain = ", ".join(f"d{index}" for index in range(rank))
    where = f" : {' and '.join(guards)}" if guards else ""
    return AffineAccess(
        isl.map(f"{isl_parameters(parameters)}{{ [{domain}] -> [{', '.join(image)}]{where} }}"),
        tuple(parameters),
    )


def window_source(
    offsets: tuple,
    rank: int,
    local: "Type",
    logical: "Type",
    carried: dict,
    extents: tuple = (),
    limits: tuple = (),
) -> "AffineAccess":
    """One operand read at its own coordinates, from where a window put them.

    A window covers the operand's shape wherever it lands, so the coordinate read
    is the one reached shifted back to where the window starts. The shift is per
    logical axis, because that is what an offset is stated against, and only then
    spread over the positions this operand's own layout made -- which are not the
    result's. An axis whose extent is given holds the read to it, because an
    operand supplying more than the window takes is not read past it; an axis
    given ``None`` is covered whole and asks for no such guard.
    """
    parameters: list[tuple[str, object]] = []
    reads: list[str] = []
    guards: list[str] = []
    for axis in range(len(logical.shape)):
        walked = carried.get(axis, "0")
        begin, bound = affine_term(offsets[axis] if axis < len(offsets) else 0, f"o{axis}")
        parameters.extend(bound)
        if bound:
            guards.append(f"0 <= {begin}")
        reads.append(walked if begin == "0" else f"{walked} - {begin}")
        if axis >= len(extents) or extents[axis] is None:
            continue
        extent, bound_extent = affine_term(extents[axis], f"e{axis}")
        parameters.extend(bound_extent)
        if bound_extent:
            guards.append(f"1 <= {extent}")
            guards.extend(_at_most(extent, limits, axis))
        guards.append(
            f"0 <= {walked} - {begin} < {extent}" if begin != "0"
            else f"0 <= {walked} < {extent}"
        )
    domain = ", ".join(f"d{index}" for index in range(rank))
    image = ", ".join(factored_image(reads, local, logical))
    where = f" : {' and '.join(guards)}" if guards else ""
    return AffineAccess(
        isl.map(f"{isl_parameters(parameters)}{{ [{domain}] -> [{image}]{where} }}"),
        tuple(parameters),
    )


def isl_parameters(parameters: list) -> str:
    """The parameter list a relation needs, or nothing when it needs none."""
    names = list(dict.fromkeys(name for name, _value in parameters))
    return f"[{', '.join(names)}] -> " if names else ""


def factored_window(
    offsets: "Sequence[object]", extents: "Sequence[object]", local: "Type", logical: "Type"
) -> tuple[tuple, tuple]:
    """Project one logical window onto the positions a layout factored.

    A window is stated per logical axis and a projected Type has one entry per
    position, so the lengths disagree the moment anything is split. A window
    covering a whole axis leaves every position of it whole. One that narrows an
    axis lands on the single position varying over it. An axis whose layout
    leaves several varying is refused: the window would have to be shown to
    align with the split first, and guessing is how a participant ends up
    charged for rows its neighbour holds.
    """
    belongs = logical_axes_of(local, logical)
    positions: dict[int, list[int]] = {}
    for position, owner in enumerate(belongs):
        positions.setdefault(owner, []).append(position)
    spread_offsets: list[object] = [0] * len(belongs)
    spread_extents: list[object] = list(local.shape)
    for axis, held in positions.items():
        offset = offsets[axis] if axis < len(offsets) else 0
        extent = extents[axis] if axis < len(extents) else None
        whole = 1
        for position in held:
            whole *= local.shape[position]
        if offset == 0 and (extent is None or extent == whole):
            continue
        carrying = [position for position in held if local.shape[position] != 1]
        if len(carrying) == 1:
            spread_offsets[carrying[0]] = offset
            spread_extents[carrying[0]] = extent if extent is not None else whole
            continue
        if True:
            raise NotImplementedError(
                f"a window of {extent!r} at {offset!r} on logical axis {axis} of "
                f"{tuple(logical.shape)} cannot be projected: its layout gives "
                f"that axis {len(carrying)} varying positions in "
                f"{tuple(local.shape)}, and "
                "the window has not been shown to align with the split"
            )
    return tuple(spread_offsets), tuple(spread_extents)


def self_image(local: "Type", logical: "Type") -> "isl.multi_aff":
    """A value read at its own coordinates, with its fixed positions written down.

    An identity over every position claims coordinates a participant does not
    have: a position it holds one of is that participant's identity, and only
    zero is in it. Writing the constant makes two names of the same bytes
    compare equal instead of differing on an axis neither can vary.
    """
    rank = len(local.shape)
    domain = ", ".join(f"d{index}" for index in range(rank))
    carried = logical_coordinates(local, logical)
    reads = [carried.get(axis, "0") for axis in range(len(logical.shape))]
    image = ", ".join(factored_image(reads, local, logical))
    if not rank:
        return isl.multi_aff("{ [] -> [] }")
    return isl.multi_aff(f"{{ [{domain}] -> [{image}] }}")


def factored_image(reads: "Sequence[str]", local: "Type", logical: "Type") -> list[str]:
    """Spread one expression per logical axis over the positions it occupies.

    A canonical `ShardLayout` factors a logical axis into several positions, and
    an image has to name every one of them or it cannot be composed with the
    `Layout` that turns positions into bytes. A position a participant holds one
    of contributes a constant, because its coordinate is the participant's
    identity rather than anything the expression varies over. The rest carry the
    expression, delinearized across them in the order the layout states.
    """
    belongs = logical_axes_of(local, logical)
    extents = list(local.shape)
    positions: dict[int, list[int]] = {}
    for position, owner in enumerate(belongs):
        positions.setdefault(owner, []).append(position)
    image = ["0"] * len(belongs)
    for owner, held in positions.items():
        carrying = [position for position in held if extents[position] != 1]
        if not carrying:
            continue
        expression = reads[owner] if owner < len(reads) else "0"
        stride = 1
        for position in reversed(carrying):
            extent = extents[position]
            walked = expression if stride == 1 else f"floor(({expression})/{stride})"
            image[position] = walked if position == carrying[0] else f"({walked}) mod {extent}"
            stride *= extent
    return image


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

    return iterating(
        getattr(result, "shape", ()) or (),
        AccessRelations(
            inputs=tuple(moves(_empty(arg), 0) for arg in call.args),
            outputs=(writes(_empty(call.args[0]) if call.args else _identity(out_rank), 0),),
        ),
    )


def linearized_view(out_shape: tuple, in_shape: tuple) -> "isl.multi_aff | isl.map":
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
    over: "Callable[..., Sequence] | None" = None,
) -> Callable[..., AccessRelations]:
    """An Op whose whole purpose is to re-address one operand's bytes.

    A reshape, a slice, a reshard, an item of a tuple: the result is those same
    elements under another name, though a name whose layout may factor its axes
    differently, so the source states its own positions. Both boundaries are
    transfers, and the output names the operand it came from through one forward
    link, so a plan that can put them at the same addresses makes this cost
    nothing and one that cannot makes it a copy. Neither is this handler's call.
    """

    def _handler(call, ctx) -> AccessRelations:
        result = ctx.local_type_of(call)
        out_rank = len(result.shape) if hasattr(result, "shape") else 0
        walked = out_rank if over is None else len(tuple(over(call, ctx)))
        moved = elements_of(result)
        held = AccessQuantity(moved, moved)

        if mapping is not None:
            reads, written = mapping(call, ctx)
        else:
            held_type = ctx.local_type_of(call.args[source])
            logical_source = ctx.type_of(call.args[source])
            logical_result = ctx.type_of(call)
            carried = (
                logical_coordinates(result, logical_result)
                if hasattr(logical_result, "shape")
                else {}
            )
            spread = (
                factored_image(
                    [
                        carried.get(axis, "0")
                        for axis in range(len(getattr(logical_result, "shape", ())))
                    ],
                    held_type,
                    logical_source,
                )
                if hasattr(held_type, "shape")
                else []
            )
            domain = ", ".join(f"d{index}" for index in range(out_rank))
            reads = (
                isl.multi_aff(f"{{ [{domain}] -> [{', '.join(spread)}] }}")
                if spread
                else _identity(walked)
            )
            written = (
                self_image(result, logical_result)
                if hasattr(logical_result, "shape")
                else _identity(out_rank)
            )
        link = StorageLink(
            kind="forward",
            input=source,
            where=reads,
            quantity=held,
            input_field=None if field is None else field(call, ctx),
        )
        return iterating(
            (getattr(result, "shape", ()) or ()) if over is None else over(call, ctx),
            AccessRelations(
                inputs=tuple(
                    BoundaryAccess(
                        reads if index == source else addresses_only(walked, ctx, arg),
                        held if index == source else AccessQuantity(0, 0),
                        AccessMode.TRANSFER if index == source else AccessMode.READ,
                    )
                    for index, arg in enumerate(call.args)
                ),
                outputs=(transfers(written, held, link),),
                storage_effect=None if storage is None else storage(call, ctx),
            ),
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

        return iterating(
            result.shape,
            AccessRelations(
                inputs=tuple(
                    moves(
                        _identity(_rank_of(call.args[index])),
                        elementwise_elements(call.args[index], call, ctx),
                    )
                    for index in range(n_inputs)
                ),
                outputs=(writes(_identity(out_rank), elements_of(result)),),
                storage_effect=None if storage is None else storage(call, ctx),
            ),
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
    "AffineAccess",
    "IndexedAccess",
    "elements_of",
    "moves",
    "placed_window",
    "moves_between",
    "OperandValue",
    "access_elements",
    "affine_term",
    "isl_parameters",
    "WindowAccess",
    "AccessRelations",
    "AccessRelationResult",
    "access_relation_registry",
    "AccessMode",
    "OutputStorage",
    "StorageLink",
    "elementwise_elements",
    "factored_image",
    "logical_axes_of",
    "logical_coordinates",
    "factored_window",
    "holds_whole_axis",
    "self_image",
    "storage_effect_of",
    "linearized_view",
    "index_set",
    "reached_elements",
    "relation_of",
    "iterating",
    "settled",
    "view_relations",
    "window_source",
    "transfers",
    "writes",
    "measures_without_reading",
    "type_relation_registry",
    "register_access_relation",
    "register_type_relation",
    "identity_relations",
    "build_relation",
]
