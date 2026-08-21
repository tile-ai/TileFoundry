"""Register the coordinates one operation reaches at each of its boundaries.

Boundary handlers return one ``BoundaryRelation`` per boundary: the relation
from the Op's own iteration space to the coordinates that value is read or
written at. Nothing else is stated. How much crosses a boundary, what an Op
walks, and whether two boundaries meet are all answers derived from those
relations, so there is one place to be right and nothing to keep in step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import isl

from tilefoundry.ir.hir._helpers import is_one
from tilefoundry.ir.types import TensorType, TupleType, Type, tensor_bytes
from tilefoundry.ir.types.dim_isl import to_dim, to_domain
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard.shard_layout import layout_axis_to_tensor_axis

from .registries import AnalysisRegistry


@dataclass(frozen=True)
class AffineAccess:
    """One boundary's relation, together with what its parameters are.

    A coordinate an Op only learns at run time is a parameter rather than a hole:
    each entry pairs the parameter's name in *relation* with the operand element
    or dimension it is, so whoever restricts the relation binds it rather than
    guessing. A relation with no parameters states none, and a function handed in
    is kept as the relation it is.
    """

    relation: "isl.map"
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.relation, isl.multi_aff):
            object.__setattr__(self, "relation", isl.map.from_multi_aff(self.relation))
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


@dataclass(frozen=True)
class BoundaryRelation:
    """One boundary, as the coordinates it reaches and nothing else.

    A relation from the Op's iteration space to that value's own coordinates is
    the whole statement. How much crossed here is what the relation reaches, so
    it is derived rather than declared alongside: two statements of one fact
    drift, and the drift is invisible until a number is wrong.
    """

    pattern: AffineAccess

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, AffineAccess):
            raise ValueError(
                f"a boundary reaches its coordinates through an AffineAccess, "
                f"which says what its parameters are; {self.pattern!r} does not"
            )


@dataclass(frozen=True)
class AccessRelations:
    """Per-Call access relations.

    One relation per boundary value, in boundary order.

    - ``inputs``: one `BoundaryRelation` per input arg, in argument order.
    - ``outputs``: one per output. Single-output ops have len 1; tuple-output
      ops have one entry per tuple field.
    """

    inputs: tuple[BoundaryRelation, ...]
    outputs: tuple[BoundaryRelation, ...]

    def __post_init__(self) -> None:
        """Refuse an impossible description here, rather than interpret it later.

        What needs the Call -- boundary count, the rank each value has -- belongs
        to the registration wrapper, which has one. What is refusable without it
        is that every boundary states a relation and that something is produced.
        """
        for side in ("inputs", "outputs"):
            stated = getattr(self, side)
            if not isinstance(stated, tuple) or not all(
                isinstance(item, BoundaryRelation) for item in stated
            ):
                raise ValueError(
                    f"{side} is one BoundaryRelation per boundary, got {stated!r}"
                )
        if not self.outputs:
            raise ValueError("an operation produces at least one value to describe")


access_relation_registry: AnalysisRegistry = AnalysisRegistry("access_relation")


def _field_of(type_: "Type", index: int) -> "Type | None":
    """One field of a tuple, or the value itself when it has no fields."""
    if isinstance(type_, TupleType):
        return type_.fields[index] if 0 <= index < len(type_.fields) else None
    return type_ if index == 0 else None


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register the one handler that states an Op's coordinates.

    The handler signature is ``(call, ctx) -> AccessRelations``. What it answers
    comes before the Call has a Type, so it may read its operands, its Op's
    attributes and the values its parameters bind, and not the Call's own Type.
    Holding that answer against the Call is a separate step, `relations_of`.
    """

    def decorate(handler: Callable) -> Callable:
        access_relation_registry.register(op_cls, handler)
        return handler

    return decorate


def coordinates_of(call, ctx) -> AccessRelations:
    """One Op's coordinates, before anything derives what the Call returns.

    This is what type inference asks, so nothing here may consult the Type being
    derived. What is held is the one thing a caller counts on without that Type:
    one boundary per operand, in argument order. Each carrier answers for its own
    parameters when it is built.
    """
    op_cls = type(call.target)
    handler = access_relation_registry.lookup(op_cls)
    if handler is None:
        raise ValueError(
            f"{op_cls.__name__} states no access relations, and there is no "
            "fallback: register one with register_access_relation"
        )
    relations = handler(call, ctx)
    if len(relations.inputs) != len(call.args):
        raise ValueError(
            f"{op_cls.__name__} describes {len(relations.inputs)} input "
            f"boundar{'y' if len(relations.inputs) == 1 else 'ies'} of a call "
            f"with {len(call.args)}"
        )
    return relations


def _affine_boundaries(relations: AccessRelations):
    """Every boundary of one Op that states coordinates, with where it is."""
    for side, boundaries in (("input", relations.inputs), ("output", relations.outputs)):
        for index, boundary in enumerate(boundaries):
            yield side, index, boundary.pattern


def iteration_universe(relations: AccessRelations) -> "isl.set | None":
    """The whole space one Op walks, from the boundaries that answer about it.

    There is no separate place an Op declares this: its boundaries do, each on
    the part it answers on, so the whole is their union once their parameters are
    lined up. Every reader derives it here so that one Op is one space to all of
    them. It says what the Op walks, not that the Op was right about it.
    """
    walked = None
    for _side, _index, pattern in _affine_boundaries(relations):
        own = relation_of(pattern).domain()
        walked = own if walked is None else walked.union(own)
    return None if walked is None else walked.coalesce()


def projected(relations: AccessRelations, call, ctx) -> AccessRelations:
    """Every boundary in the coordinates the reader asking can address.

    An Op states where it reads and writes among logical axes, because that is
    all it can know before anything is placed. A reader addresses positions, and
    which ones a logical coordinate is depends on the layout the value ended up
    with, so the two are composed here for every Op. That composition also holds
    a participant to its own iterations, and every boundary is then held to the
    same ones: a value nobody sharded is addressed whole by everyone, so left
    alone it would charge one participant the whole of what all of them read.
    """
    held = ctx.local_type_of(call)
    fields = held.fields if isinstance(held, TupleType) else (held,)
    logical = ctx.type_of(call)
    logical_fields = logical.fields if isinstance(logical, TupleType) else (logical,)
    bindings = parameters_of(relations)

    def views(index: int, side: str) -> tuple:
        if side == "output":
            return (
                fields[index] if index < len(fields) else None,
                logical_fields[index] if index < len(logical_fields) else None,
            )
        arg = call.args[index]
        return ctx.local_type_of(arg), ctx.type_of(arg)

    placed = {
        side: tuple(
            _placed(boundary, *views(index, side), side, index, call)
            for index, boundary in enumerate(boundaries)
        )
        for side, boundaries in (("input", relations.inputs), ("output", relations.outputs))
    }
    answered = {
        side: tuple(
            _answered(boundary, views(index, side)[1])
            for index, boundary in enumerate(boundaries)
        )
        for side, boundaries in (("input", relations.inputs), ("output", relations.outputs))
    }
    share = _own_iterations(relations, placed, answered)
    carried = {
        side: tuple(
            _addressed(relation, views(index, side)[0], bindings, side, index, call)
            for index, relation in enumerate(relations_placed)
        )
        for side, relations_placed in placed.items()
    }
    if share is None:
        return AccessRelations(inputs=carried["input"], outputs=carried["output"])
    return AccessRelations(
        inputs=tuple(
            _iterating_over(boundary, share, bindings) for boundary in carried["input"]
        ),
        outputs=tuple(
            _iterating_over(boundary, share, bindings) for boundary in carried["output"]
        ),
    )


def _answered(boundary: "BoundaryRelation", logical) -> "isl.set":
    """Where a boundary answers about coordinates the value actually has.

    A relation may be written to reach past its value -- a window shifted back
    to where it came from does -- and outside that it is saying nothing rather
    than saying this participant does not iterate there.
    """
    relation = relation_of(boundary.pattern)
    box = index_set(tuple(logical.shape)) if isinstance(logical, TensorType) else None
    if box is None or box.tuple_dim() != relation.range().tuple_dim():
        return relation.domain()
    return relation.intersect_range(box).domain()


def _own_iterations(
    stated: AccessRelations, placed: dict, answered: dict
) -> "isl.set | None":
    """Which of an Op's iterations this participant performs, or None if all.

    A value handed out in pieces says which iterations belong to whoever holds
    this piece, and says it only where its boundary answers at all: outside
    that, reading its silence as a restriction cuts away the very part another
    boundary is there to describe. So each allows what it owns together with
    everything it was not asked about, and one that reaches nothing allows all.
    What a parameter may be travels with the answer.
    """
    walked = iteration_universe(stated)
    if walked is None:
        return None
    share = None
    limits = None
    for side, boundaries in (("input", stated.inputs), ("output", stated.outputs)):
        for index, boundary in enumerate(boundaries):
            asked = relation_of(boundary.pattern)
            if asked.is_empty():
                continue
            try:
                allowed = placed[side][index].domain().union(
                    walked.subtract(answered[side][index])
                )
                bounds = asked.params()
            except isl.Error as error:
                raise ValueError(
                    f"{side} {index} cannot be lined up with the space its Op "
                    f"walks, so which iterations are this participant's is not "
                    f"answerable: {error}"
                ) from error
            share = allowed if share is None else share.intersect(allowed)
            limits = bounds if limits is None else limits.intersect(bounds)
    if share is None:
        return None
    share = share.intersect(walked)
    if limits is not None:
        share = share.intersect_params(limits)
    share = share.coalesce()
    return None if share.is_equal(walked) else share


def _iterating_over(
    boundary: "BoundaryRelation", share: "isl.set", bindings: dict
) -> "BoundaryRelation":
    """One boundary held to the iterations its participant performs.

    Restricting can bring in a parameter another boundary named, so what they
    stand for comes from the whole Op rather than from this boundary alone.
    """
    relation = relation_of(boundary.pattern)
    try:
        held = relation.intersect_domain(share)
    except isl.Error as error:
        raise ValueError(
            f"a boundary at {relation} cannot be held to the iterations "
            f"{share} this participant performs: {error}"
        ) from error
    return _rebuilt(held, bindings)


def renaming_relation(call, ctx) -> "AffineAccess":
    """One view's own coordinates, as coordinates of the value it renames.

    A view states where it reads and where it writes over one space, so going
    from its result's coordinates to its source's is reading the second
    backwards and the first forwards. Every consumer that folds a view into its
    buffer asks for this rather than rebuilding the Op's arithmetic, which is
    how one relation answers dependence, footprint and movement alike.
    """
    relations = relations_of(call, ctx)
    if isinstance(ctx.type_of(call.args[0]), TupleType):
        raise ValueError(
            f"{type(call.target).__name__} renames a field of a tuple, which is "
            "one leaf of it rather than a coordinate change to fold"
        )
    written = relation_of(relations.outputs[0].pattern)
    reads = relation_of(relations.inputs[0].pattern)
    folded = written.reverse().apply_range(reads)
    bindings = parameters_of(relations)
    return AffineAccess(
        folded,
        tuple(
            (name, bindings[name])
            for name in (
                folded.get_dim_name(isl.dim_type.PARAM, index)
                for index in range(folded.dim(isl.dim_type.PARAM))
            )
            if name in bindings
        ),
    )


def parameters_of(relations: AccessRelations) -> dict:
    """Every parameter this Op binds, by name, across all of its boundaries.

    One name is one value for the whole Op, so a relation that gains a parameter
    by being composed or restricted still knows what it stands for. This is also
    what decodes an extent back out of the space an Op walks: a symbolic axis is
    a parameter there, and this says which dimension it was.
    """
    bindings: dict = {}
    for _side, _index, pattern in _affine_boundaries(relations):
        bindings.update(getattr(pattern, "parameters", ()) or ())
    return bindings


def shape_from_relation(
    relations: AccessRelations, extents: "Sequence", *, output: int = 0
) -> tuple:
    """The extents one output reaches, which is the shape that output has.

    Type inference and every other reader take the shape from the same relation,
    so a relation that contracts the wrong axis is wrong for all of them rather
    than for whichever one recomputed it. *extents* is what the Op walks: an
    empty space reaches nothing and has no extent left to read, so a projected
    axis takes its own from there in order.
    """
    reached = relation_of(relations.outputs[output].pattern)
    rank = reached.dim(isl.dim_type.OUT)
    if reached.is_empty():
        return tuple(extents[axis] for axis in range(rank))
    image = reached.range()
    bindings = parameters_of(relations)
    return tuple(
        to_dim(image.dim_max(axis).add_constant(1), bindings) for axis in range(rank)
    )


def boundary_maps(relations: AccessRelations) -> tuple["isl.map", ...]:
    """Every boundary's relation, inputs first and then outputs.

    Boundary order is already the argument order, so a reader that wants the
    Op's maps as one sequence -- shard propagation walks operands against the
    result -- takes them here rather than knowing how the record is shaped.
    """
    return tuple(
        relation_of(boundary.pattern)
        for boundary in (*relations.inputs, *relations.outputs)
    )


def _placed(
    boundary: "BoundaryRelation", local, logical, side: str, index: int, call
) -> "isl.map":
    """One boundary's image carried from logical axes onto the positions it has.

    Held to the positions this participant was given, so which iterations are
    its own follows from the placement rather than from a relation that may
    reach past what it was handed.
    """
    relation = relation_of(boundary.pattern)
    if (
        not isinstance(local, TensorType)
        or not isinstance(logical, TensorType)
        or relation.is_empty()
    ):
        return relation
    if relation.dim(isl.dim_type.OUT) != len(logical.shape):
        raise ValueError(
            f"{type(call.target).__name__} reads {side} {index} at "
            f"{relation.dim(isl.dim_type.OUT)} coordinates, and that value has "
            f"{len(logical.shape)} axes of its own; a canonical relation is "
            "stated in the axes an Op was written in"
        )
    return _within_positions(relation.apply_range(positions_of(local, logical)), local)


def _within_positions(relation: "isl.map", local) -> "isl.map":
    """One relation held to the coordinates the value it reaches actually has."""
    box = index_set(tuple(local.shape)) if isinstance(local, TensorType) else None
    if box is None or box.tuple_dim() != relation.range().tuple_dim():
        return relation
    return relation.intersect_range(box)


def _addressed(
    relation: "isl.map", local, bindings: dict, side: str, index: int, call
) -> "BoundaryRelation":
    """One placed boundary, held to the coordinates the value actually has.

    The projected relation is then the whole answer: what it reaches is what
    crossed, with nothing left for a reader to intersect again or to forget to.
    """
    return _held_countable(
        _rebuilt(_within_positions(relation, local), bindings), side, index, call
    )


def _rebuilt(relation: "isl.map", bindings: dict) -> "BoundaryRelation":
    """One boundary carrying a relation, with what its parameters stand for."""
    return BoundaryRelation(
        AffineAccess(
            relation,
            tuple(
                (name, bindings[name])
                for name in (
                    relation.get_dim_name(isl.dim_type.PARAM, index)
                    for index in range(relation.dim(isl.dim_type.PARAM))
                )
                if name in bindings
            ),
        )
    )


def _held_countable(
    boundary: "BoundaryRelation", side: str, index: int, call
) -> "BoundaryRelation":
    """Refuse a projected boundary nobody can count.

    A relation is the only statement of how much crosses here, so one whose
    image is not a number leaves a reader with nothing -- and there is no
    falling back on what the Op said, because two answers is the thing this
    carrier exists to remove.
    """
    if reached_elements(boundary.pattern) is None:
        raise ValueError(
            f"{type(call.target).__name__} states {side} {index} as "
            f"{relation_of(boundary.pattern)}, which reaches no countable number "
            "of elements here"
        )
    return boundary


def relations_of(call, ctx) -> AccessRelations:
    """One Op's coordinates, held also against the Type the Call now has.

    The Op stated its coordinates in its own axes; here they are carried onto the
    positions this reader addresses, and then held to the Call. What is checked
    is what needs the Type: one boundary per output field, each written at the
    rank that field has in this view.
    """
    op_cls = type(call.target)
    relations = projected(coordinates_of(call, ctx), call, ctx)
    result = ctx.local_type_of(call)
    wanted = len(result.fields) if isinstance(result, TupleType) else 1
    if len(relations.outputs) != wanted:
        raise ValueError(
            f"{op_cls.__name__} describes {len(relations.outputs)} output "
            f"boundar{'y' if len(relations.outputs) == 1 else 'ies'} of a call "
            f"with {wanted}"
        )
    return relations


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


def normalised_rows(local: "Type", logical: "Type", first: int) -> tuple:
    """The rows an Op is asked once per, and the names its reads range over.

    Normalising needs the whole of what it normalises before any of it can be
    written, so the axes from `first` on are not coordinates the Op is asked by:
    they are free in the images. Returns the extents walked, one name per
    position of `local`, and the guards bounding the free ones.
    """
    belongs = logical_axes_of(local, logical)
    extents: list = []
    names: list[str] = []
    guards: list[str] = []
    for position, owner in enumerate(belongs):
        extent = local.shape[position]
        if owner < first:
            names.append(f"d{len(extents)}")
            extents.append(extent)
        elif is_one(extent):
            names.append("0")
        else:
            names.append(f"j{position}")
            guards.append(f"0 <= j{position} < {extent}")
    return tuple(extents), tuple(names), tuple(guards)


def logical_term(names: "Sequence[str]", local: "Type", logical: "Type", axis: int) -> str:
    """One logical axis's coordinate, rebuilt from the positions holding it."""
    linear, stride = "", 1
    belongs = logical_axes_of(local, logical)
    for position in reversed(range(len(belongs))):
        extent = local.shape[position]
        if belongs[position] != axis or is_one(extent):
            continue
        term = names[position] if stride == 1 else f"{stride} * {names[position]}"
        linear = term if not linear else f"{linear} + {term}"
        stride *= extent
    return linear or "0"


def iterating(extents: "Sequence", relations: "AccessRelations") -> "AccessRelations":
    """Every boundary of one Op, on the iteration space that Op walks.

    An access map's domain is the Op's whole iteration space, so its bounds are
    the Op's and every boundary shares them -- not each boundary's own, and not
    a reader's guess from a result's rank. A contraction walks the axis it
    contracts; most Ops walk what they produce. A boundary may be partial in
    that space, which is one relation empty somewhere, not a second space.
    """
    try:
        domain, named = to_domain(tuple(extents))
    except (TypeError, ValueError, isl.Error) as error:
        raise ValueError(
            f"an Op states it iterates {tuple(extents)}, which is no space to "
            f"walk: {error}"
        ) from error
    return AccessRelations(
        inputs=tuple(_held_to(boundary, domain, named) for boundary in relations.inputs),
        outputs=tuple(_held_to(boundary, domain, named) for boundary in relations.outputs),
    )


def _held_to(
    boundary: "BoundaryRelation", domain: "isl.set", named: dict
) -> "BoundaryRelation":
    """One boundary, restricted to the coordinates its Op iterates."""
    pattern = boundary.pattern
    relation = relation_of(pattern)
    if relation.dim(isl.dim_type.IN) != domain.dim(isl.dim_type.SET):
        raise ValueError(
            f"a boundary is asked by {relation.dim(isl.dim_type.IN)} coordinates "
            f"and its Op iterates {domain.dim(isl.dim_type.SET)}; one Op states "
            "one coordinate system and every boundary of it answers about that one"
        )
    held = relation.intersect_domain(domain)
    stated = dict(pattern.parameters) if isinstance(pattern, AffineAccess) else {}
    return BoundaryRelation(
        AffineAccess(
            held,
            tuple(
                (name, stated.get(name, named.get(name)))
                for name in (
                    held.get_dim_name(isl.dim_type.PARAM, index)
                    for index in range(held.dim(isl.dim_type.PARAM))
                )
            ),
        )
    )


def relation_of(pattern: "AffineAccess") -> "isl.map":
    """One boundary's coordinates as the relation it states."""
    if not isinstance(pattern, AffineAccess):
        raise ValueError(f"a boundary states an AffineAccess, not {pattern!r}")
    return pattern.relation


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


def settled(pattern: "AffineAccess") -> "isl.map":
    """One relation with every parameter fixed to a number.

    A parameter bound to something that has a value is fixed to that value. One
    whose value nobody here holds is fixed to the smallest the relation itself
    allows: the first legal iteration of a loop, the smallest legal window of a
    runtime extent. Either way a reader gets a number it can check rather than a
    range it has to interpret, so the parameters leave with their values put in.
    A relation nothing satisfies has no value to settle on.
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
            edge = str(probe.dim_min_val(0))
            try:
                number = int(edge)
            except ValueError:
                raise ValueError(
                    f"a relation leaves {name!r} at {edge} because it does not "
                    "state what that parameter may be; a boundary nobody can bind "
                    "is not one a reader can count"
                ) from None
        legal = legal.intersect(isl.set(f"{space}{{ : {name} = {number} }}"))
    settled_at = relation.intersect_params(legal)
    return settled_at.project_out(isl.dim_type.PARAM, 0, len(names))


def reached_elements(
    pattern: "AffineAccess", box: "isl.set | None" = None, within: "isl.set | None" = None
) -> int | None:
    """How many distinct boundary elements one boundary reaches.

    Reaching the same element from many coordinates is one element moved, not
    many dependences, so an inner iteration axis costs nothing; and reaching
    past the coordinates the operand has is not reaching at all. This is one
    occurrence, so a parameter nobody bound settles at its first legal binding:
    how many crossings a loop performs is the footprint family's question.
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


def control_leaves(type_: "Type") -> int:
    """How many numbers one operand carries for placing or sizing a window.

    A window is placed by one number per axis it is placed on, and an operand
    holding several of them carries several: a tuple of offsets is read once per
    leaf it holds, however its fields are nested.
    """
    return len(leaves_of(type_))


def leaves_of(type_: "Type") -> tuple:
    """Every tensor leaf of one value, flat, in the order a reader indexes them."""
    if isinstance(type_, TupleType):
        return tuple(leaf for field in type_.fields for leaf in leaves_of(field))
    return (type_,) if isinstance(type_, TensorType) else ()


def leaf_span(type_: "Type", field: int) -> "tuple[int, int]":
    """Where one field of a value begins among its flat leaves, and how many.

    A structured value is indexed by leaf, not by top-level field, so a field
    holding a tuple of its own covers a run of them. Whoever takes that field
    takes that run.
    """
    if not isinstance(type_, TupleType):
        return (0, len(leaves_of(type_)))
    begin = sum(len(leaves_of(held)) for held in type_.fields[:field])
    return (begin, len(leaves_of(type_.fields[field])))


def reached_leaves(pattern: "AffineAccess", count: int) -> "frozenset[int] | None":
    """Which of a structured value's flat leaves one boundary reaches.

    A tuple of numbers is indexed by one coordinate, so what crosses there is a
    set of leaves rather than a count: they need not be the same width, and
    charging the first for the one that was taken is a wrong number at the right
    size. Read at the same first legal binding one crossing is counted at.
    """
    image = settled(pattern).range()
    if image.dim(isl.dim_type.PARAM) or image.tuple_dim() != 1:
        return None
    return frozenset(
        leaf
        for leaf in range(count)
        if not image.intersect(isl.set(f"{{ [{leaf}] }}")).is_empty()
    )


def _control_space(rank: int, ctx, arg) -> "tuple[str, str, str]":
    """The domain, image and reach of one control operand's own coordinates.

    A tuple of numbers is indexed flat, one leaf however its fields are nested.
    A lone scalar's legal index set is the single point, at whatever positions
    its own view gives it.
    """
    domain = ", ".join(f"d{index}" for index in range(rank))
    stated = ctx.type_of(arg)
    if isinstance(stated, TupleType):
        return domain, "l", f"0 <= l < {control_leaves(stated)}"
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


def positions_of(local: "Type", logical: "Type") -> "isl.map":
    """Where one value's logical coordinates live among the positions it has.

    A layout may factor a logical axis into several positions, and then a
    coordinate on that axis is the mixed-radix digits of those positions: the
    outer ones are what it divides by, the inner ones what it is left with.
    Composing an Op's logical relation with this is how a reader gets the
    coordinates it can address, without any Op saying how. An axis nobody
    divided is left unguarded, holding all of it saying nothing about whose
    iterations are whose.
    """
    belongs = logical_axes_of(local, logical)
    coordinates = ", ".join(f"c{axis}" for axis in range(len(logical.shape)))
    image = factored_image(
        [f"c{axis}" for axis in range(len(logical.shape))], local, logical
    )
    guards = []
    held: dict[int, int] = {}
    for position, owner in enumerate(belongs):
        extent = local.shape[position]
        if isinstance(extent, int) and not isinstance(extent, bool):
            held[owner] = held.get(owner, 1) * extent
    for axis, extent in sorted(held.items()):
        whole = logical.shape[axis] if axis < len(logical.shape) else None
        if isinstance(whole, int) and not isinstance(whole, bool) and extent >= whole:
            continue
        guards.append(f"0 <= c{axis} < {extent}")
    where = f" : {' and '.join(guards)}" if guards else ""
    return isl.map(f"{{ [{coordinates}] -> [{', '.join(image)}]{where} }}")


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


def identity_access(rank: int) -> "AffineAccess":
    """Identity boundary for a tensor of *rank*: each element read where it is."""
    dims = ", ".join(f"d{index}" for index in range(rank))
    return AffineAccess(isl.map(f"{{ [{dims}] -> [{dims}] }}" if rank else "{ [] -> [] }"))


def broadcast_access(result_shape: tuple, operand_shape: tuple) -> "AffineAccess":
    """Which coordinate of an operand a result coordinate reads.

    An operand of the result's own shape reads the coordinate it is at. A
    shorter one right-aligns, dropping the leading axes it does not have; an
    axis it holds one of is read at zero however far the result runs along it.
    Both are still functions of the result coordinate, and both are stated as
    the relation they are.
    """
    rank = len(result_shape)
    dims = [f"d{index}" for index in range(rank)]
    offset = rank - len(operand_shape)
    reads = [
        "0" if operand_shape[index - offset] == 1 else dims[index]
        for index in range(offset, rank)
    ]
    domain = ", ".join(dims)
    if not reads:
        return AffineAccess(isl.map(f"{{ [{domain}] -> [] }}" if rank else "{ [] -> [] }"))
    return AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def measures_without_reading(call, ctx) -> AccessRelations:
    """An Op that answers from a value's Type rather than from its elements.

    A rank, a shape, a name for the same value at another level: the answer is
    already in the Type, so no coordinate is read. The relation says that -- an
    empty map, nothing crossing -- rather than an identity claiming a read the
    Op never performs.
    """
    result = ctx.type_of(call.args[0]) if call.args else None
    out_rank = len(result.shape) if hasattr(result, "shape") else 0

    def _empty(arg) -> "AffineAccess":
        type_ = ctx.type_of(arg)
        in_rank = len(type_.shape) if hasattr(type_, "shape") else 0
        reads = ", ".join(f"i{index}" for index in range(in_rank))
        domain = ", ".join(f"d{index}" for index in range(out_rank))
        return AffineAccess(isl.map(f"{{ [{domain}] -> [{reads}] : 1 = 0 }}"))

    return iterating(
        getattr(result, "shape", ()) or (),
        AccessRelations(
            inputs=tuple(BoundaryRelation(_empty(arg)) for arg in call.args),
            outputs=(
                BoundaryRelation(
                    _empty(call.args[0]) if call.args else identity_access(out_rank)
                ),
            ),
        ),
    )


def linearized_view(out_shape: tuple, in_shape: tuple) -> "AffineAccess":
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
        return AffineAccess(isl.map(f"{{ [{dims}] -> [{reads}] : 1 = 0 }}"))
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
    return AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def view_relations(
    source: int = 0,
    mapping: "Callable[..., tuple[AffineAccess, AffineAccess]] | None" = None,
    field: "Callable[..., int | None] | None" = None,
    over: "Callable[..., Sequence] | None" = None,
) -> Callable[..., AccessRelations]:
    """An Op whose whole purpose is to re-address one operand's bytes.

    A reshape, a slice, a reshard, an item of a tuple: the result is those same
    elements under another name, though a name whose layout may factor its axes
    differently, so the source states its own positions. Where those elements
    came from is all this states; whether the two ends can be given the same
    addresses is the allocation's answer, not this handler's.
    """

    def _handler(call, ctx) -> AccessRelations:
        held = ctx.type_of(call.args[source])
        taken = 0 if field is None else (field(call, ctx) or 0)
        result = _field_of(held, taken)
        out_rank = len(result.shape) if hasattr(result, "shape") else 0
        walked = out_rank if over is None else len(tuple(over(call, ctx)))
        out_rank = walked

        if mapping is not None:
            reads, written = mapping(call, ctx)
        else:
            reads = identity_access(walked)
            written = identity_access(out_rank)
        if isinstance(held, TupleType):
            begin, count = leaf_span(held, taken)
            coordinates = ", ".join(f"d{index}" for index in range(walked))
            reads = AffineAccess(
                isl.map(
                    f"{{ [{coordinates}] -> [l] : {begin} <= l < {begin + count} }}"
                )
            )
        walks = (getattr(result, "shape", ()) or ()) if over is None else over(call, ctx)
        return iterating(
            walks,
            AccessRelations(
                inputs=tuple(
                    BoundaryRelation(
                        reads if index == source else control_read(walked, ctx, arg)
                    )
                    for index, arg in enumerate(call.args)
                ),
                outputs=(BoundaryRelation(written),),
            ),
        )

    return _handler


def identity_relations(n_inputs: int) -> Callable[..., AccessRelations]:
    """Identity relations.

    Factory for a GLOBAL-level access-relation handler whose ``n_inputs``
    inputs and single output are all elementwise identity.

    Each input contributes its own-rank identity; the output uses its own
    rank. A structural (non-tensor) input arg — e.g. ``TupleGetItem``'s tuple
    operand — has no shape of its own, so it borrows the output's rank.
    """

    def _handler(call, ctx) -> AccessRelations:
        walked = ctx.type_of(call.args[0])
        out_rank = len(walked.shape)

        def _rank_of(arg) -> int:
            ty = ctx.type_of(arg)
            return len(ty.shape) if hasattr(ty, "shape") else out_rank

        return iterating(
            walked.shape,
            AccessRelations(
                inputs=tuple(
                    BoundaryRelation(identity_access(_rank_of(call.args[index])))
                    for index in range(n_inputs)
                ),
                outputs=(BoundaryRelation(identity_access(out_rank)),),
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


__all__ = [
    "AccessRelations",
    "AffineAccess",
    "BoundaryRelation",
    "access_relation_registry",
    "coordinates_of",
    "index_set",
    "iterating",
    "identity_access",
    "identity_relations",
    "logical_axes_of",
    "logical_coordinates",
    "placed_window",
    "boundary_maps",
    "projected",
    "leaves_of",
    "reached_elements",
    "reached_leaves",
    "register_access_relation",
    "relation_of",
    "relations_of",
    "renaming_relation",
    "settled",
    "shape_from_relation",
    "static_bytes",
    "view_relations",
    "window_source",
]
