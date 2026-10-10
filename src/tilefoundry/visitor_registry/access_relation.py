"""Register the coordinates one operation reaches at each of its boundaries.

Handlers return one ``AccessRelation`` per boundary, in one flat tuple: one per
argument in argument order, then one per result field. Each is the relation
from the Op's own iteration space to the coordinates that value is read or
written at. Nothing else is stated. How much crosses a boundary, what an Op
walks, and whether two boundaries meet are all answers derived from those
relations, so there is one place to be right and nothing to keep in step.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable

import isl

from tilefoundry.ir.core.expr import Constant
from tilefoundry.ir.isl_interop import (
    IslParamValues,
    isl_to_dim,
    layout_to_isl_map,
    shape_to_isl_set,
)
from tilefoundry.ir.types import TensorType, TupleType, Type
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import layout_axis_to_tensor_axis, shard_layout_of
from tilefoundry.ir.types.utils import (
    divided_mesh_axes,
    is_literal_shape,
    static_dim_value,
    tensor_bytes,
)
from tilefoundry.utils.isl_utils import cardinality

from .registries import DispatchRegistry


@dataclass(frozen=True)
class AccessRelation:
    """One boundary's relation, together with what its parameters are.

    A coordinate an Op only learns at run time is a parameter rather than a hole:
    *values* maps each parameter's name in *relation* to the operand element or
    dimension it is, so whoever restricts the relation binds it rather than
    guessing. A relation with no parameters states none, and a function handed in
    is kept as the relation it is. How much crossed here is what the relation
    reaches, so it is derived rather than declared alongside.
    """

    relation: "isl.map"
    values: IslParamValues = field(default_factory=dict)
    lookup: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.relation, isl.multi_aff):
            object.__setattr__(self, "relation", isl.map.from_multi_aff(self.relation))
        if not isinstance(self.relation, isl.map):
            raise ValueError(f"an access relation is a relation, not {self.relation!r}")
        object.__setattr__(self, "values", dict(self.values))
        named = set(_parameter_names(self.relation))
        if named != set(self.values):
            raise ValueError(
                f"an access relation binds {sorted(self.values)} but its relation names "
                f"{sorted(named)}; a parameter nobody can bind is a hole"
            )


def _parameter_names(relation) -> list[str]:
    """The parameters *relation* names, in its own order."""
    return [
        relation.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(relation.dim(isl.dim_type.PARAM))
    ]


access_relation_registry: DispatchRegistry = DispatchRegistry("access_relation")


def _field_of(type_: "Type", index: int) -> "Type | None":
    """One field of a tuple, or the value itself when it has no fields."""
    if isinstance(type_, TupleType):
        return type_.fields[index] if 0 <= index < len(type_.fields) else None
    return type_ if index == 0 else None


def register_access_relation(op_cls: type) -> Callable[[Callable], Callable]:
    """Decorator to register the one handler that states an Op's coordinates.

    The handler signature is ``(call, ctx) -> tuple[AccessRelation, ...]``: one
    per argument, then one per result field. It answers before the Call has a
    Type, so it may read its operands, its Op's attributes and the values its
    parameters bind, and not the Call's own Type. Projecting that answer onto a
    reader's view is a separate step, `local_relations_of`.
    """

    def decorate(handler: Callable) -> Callable:
        access_relation_registry.register(op_cls, handler)
        return handler

    return decorate


def relations_of(call, ctx) -> tuple[AccessRelation, ...]:
    """One Op's declared relations, before deriving what the Call returns.

    This is what type inference asks, so nothing here may consult the Type being
    derived. What is held is the one thing a caller counts on without that Type:
    one relation per operand, in argument order, and then at least one result.
    Each relation answers for its own parameters when it is built.
    """
    op_cls = type(call.target)
    handler = access_relation_registry.lookup(op_cls)
    if handler is None:
        raise ValueError(
            f"{op_cls.__name__} states no access relations, and there is no "
            "fallback: register one with register_access_relation"
        )
    relations = handler(call, ctx)
    if not isinstance(relations, tuple) or not all(
        isinstance(access, AccessRelation) for access in relations
    ):
        raise ValueError(
            f"{op_cls.__name__} states its boundaries as one AccessRelation each, "
            f"in a tuple; got {relations!r}"
        )
    if len(relations) <= len(call.args):
        raise ValueError(
            f"{op_cls.__name__} describes {len(relations)} boundar"
            f"{'y' if len(relations) == 1 else 'ies'} of a call with {len(call.args)} "
            "arguments; one per argument comes first, and then what it produces"
        )
    return relations


def _boundary(index: int, ninputs: int) -> str:
    """Where one relation of an Op's tuple is, for a message."""
    return f"input {index}" if index < ninputs else f"output {index - ninputs}"


def iteration_universe(relations: tuple[AccessRelation, ...]) -> "isl.set | None":
    """The whole space one Op walks, from the boundaries that answer about it.

    There is no separate place an Op declares this: its boundaries do, each on
    the part it answers on, so the whole is their union once their parameters are
    lined up. Every reader derives it here so that one Op is one space to all of
    them. It says what the Op walks, not that the Op was right about it.
    """
    walked = None
    for access in relations:
        own = access.relation.domain()
        walked = own if walked is None else walked.union(own)
    return None if walked is None else walked.coalesce()


def projected(
    relations: tuple[AccessRelation, ...],
    call,
    ctx,
    *,
    device: "Mapping[str, int] | None" = None,
) -> tuple[AccessRelation, ...]:
    """Every boundary in the coordinates the reader asking can address.

    An Op states where it reads and writes among logical axes; a reader addresses
    positions, which depend on the layout the value ended up with, so the two
    are composed here for every Op. That also holds a participant to its own
    iterations, and every boundary to the same ones. With no topology level the
    coordinates stay logical; with one, they are the positions the unit
    *device* holds -- a coordinate per mesh axis, keyed by its name (or number
    when unnamed), each 0 unless given. The arguments' relations come first.
    """
    ninputs = len(call.args)
    held = ctx.local_type_of(call)
    fields = held.fields if isinstance(held, TupleType) else (held,)
    logical = ctx.type_of(call)
    logical_fields = logical.fields if isinstance(logical, TupleType) else (logical,)
    if len(relations) - ninputs != len(fields):
        raise ValueError(
            f"{type(call.target).__name__} describes {len(relations) - ninputs} output "
            f"boundar{'y' if len(relations) - ninputs == 1 else 'ies'} of a call "
            f"with {len(fields)}"
        )
    bindings = _values_of(relations)
    level = getattr(ctx, "topology_level", None)
    coordinates: IslParamValues = {}

    def placement(value) -> "isl.map | None":
        shard = shard_layout_of(getattr(value, "layout", None))
        if level is None or not isinstance(value, TensorType) or shard is None:
            return None
        placed_at = layout_to_isl_map(
            tuple(value.shape),
            value.layout,
            coordinates,
            divided=lambda layer: divided_mesh_axes(
                layer, topology_level=level, topologies=ctx.topologies
            ),
        )
        return _at_device(placed_at, coordinates, device or {})

    views = (
        *((ctx.local_type_of(arg), ctx.type_of(arg)) for arg in call.args),
        *zip(fields, logical_fields, strict=True),
    )
    where = tuple(_boundary(index, ninputs) for index in range(len(relations)))
    placed = tuple(
        _placed(access, *view, label, call, placement, bindings)
        for access, view, label in zip(relations, views, where, strict=True)
    )
    answered = tuple(
        _answered(access, view[1]) for access, view in zip(relations, views, strict=True)
    )
    share = _own_iterations(relations, placed, answered, ninputs)
    carried = tuple(
        _addressed(relation, view[0], bindings, label, call, lookup=access.lookup)
        for relation, view, label, access in zip(placed, views, where, relations, strict=True)
    )
    if share is None:
        return carried
    return tuple(_iterating_over(access, share, bindings) for access in carried)


def _answered(access: AccessRelation, logical) -> "isl.set":
    """Where a boundary answers about coordinates the value actually has.

    A relation may be written to reach past its value -- a window shifted back
    to where it came from does -- and outside that it is saying nothing rather
    than saying this participant does not iterate there.
    """
    relation = access.relation
    if not isinstance(logical, TensorType) or not is_literal_shape(logical.shape):
        return relation.domain()
    box = shape_to_isl_set(tuple(logical.shape), {})
    if box.tuple_dim() != relation.range().tuple_dim():
        return relation.domain()
    return relation.intersect_range(box).domain()


def _own_iterations(
    stated: tuple[AccessRelation, ...], placed: tuple, answered: tuple, ninputs: int
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
    for index, access in enumerate(stated):
        asked = access.relation
        if asked.is_empty():
            continue
        try:
            allowed = placed[index].domain().union(walked.subtract(answered[index]))
            bounds = asked.params()
        except isl.Error as error:
            raise ValueError(
                f"{_boundary(index, ninputs)} cannot be lined up with the space its "
                f"Op walks, so which iterations are this participant's is not "
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
    access: AccessRelation, share: "isl.set", bindings: IslParamValues
) -> AccessRelation:
    """One boundary held to the iterations its participant performs.

    Restricting can bring in a parameter another boundary named, so what they
    stand for comes from the whole Op rather than from this boundary alone.
    """
    try:
        held = access.relation.intersect_domain(share)
    except isl.Error as error:
        raise ValueError(
            f"a boundary at {access.relation} cannot be held to the iterations "
            f"{share} this participant performs: {error}"
        ) from error
    names = _parameter_names(held)
    return AccessRelation(
        held, {name: bindings[name] for name in names if name in bindings}, access.lookup
    )


def renaming_relation(call, ctx, local_relations: tuple[AccessRelation, ...]) -> AccessRelation:
    """One view's own coordinates, as coordinates of the value it renames.

    A view states where it reads and where it writes over one space, so going
    from its result's coordinates to its source's is reading the second
    backwards and the first forwards. Every consumer that folds a view into its
    buffer asks for this rather than rebuilding the Op's arithmetic, which is
    how one relation answers dependence, footprint and movement alike.
    ``local_relations`` is the call's relations projected into ``ctx``'s window.
    """
    if isinstance(ctx.type_of(call.args[0]), TupleType):
        raise ValueError(
            f"{type(call.target).__name__} renames a field of a tuple, which is "
            "one leaf of it rather than a coordinate change to fold"
        )
    written = local_relations[len(call.args)].relation
    reads = local_relations[0].relation
    folded = written.reverse().apply_range(reads)
    bindings = _values_of(local_relations)
    names = _parameter_names(folded)
    return AccessRelation(
        folded,
        {name: bindings[name] for name in names if name in bindings},
        local_relations[0].lookup,
    )


def _values_of(relations: tuple[AccessRelation, ...]) -> IslParamValues:
    """Every parameter this Op binds, by name, across all of its boundaries.

    One name is one value for the whole Op, so a relation that gains a parameter
    by being composed or restricted still knows what it stands for, and a name
    two boundaries bind to different values is refused rather than resolved by
    whichever came last.
    """
    merged: IslParamValues = {}
    for access in relations:
        for name, value in access.values.items():
            if merged.setdefault(name, value) is not value:
                raise ValueError(
                    f"parameter {name!r} stands for two different values across one "
                    f"Op's boundaries; one name in one Op is one value"
                )
    return merged


def shape_from_relation(access: AccessRelation, extents: "Sequence") -> tuple:
    """The extents one result reaches, which is the shape that result has.

    Type inference and every other reader take the shape from the same relation,
    so a relation that contracts the wrong axis is wrong for all of them rather
    than for whichever one recomputed it. *extents* is what the Op walks: an
    empty space reaches nothing and has no extent left to read, so a projected
    axis takes its own from there in order. A symbolic extent is a parameter of
    *access*, and its own values say which dimension it was.
    """
    reached = access.relation
    rank = reached.dim(isl.dim_type.OUT)
    if reached.is_empty():
        return tuple(extents[axis] for axis in range(rank))
    image = reached.range()
    return tuple(
        isl_to_dim(image.dim_max(axis).add_constant(1), access.values) for axis in range(rank)
    )


def _at_device(placement: "isl.map", coordinates: IslParamValues, device: Mapping) -> "isl.map":
    """*placement* at one unit: each mesh coordinate fixed, then gone.

    A coordinate is fixed to the value *device* gives under its mesh axis's
    name, or 0. A value that is not one of that axis's positions is refused,
    rather than fixing the unit nowhere and counting nothing.
    """
    for name in _parameter_names(placement):
        target = coordinates[name].target
        axis = static_dim_value(coordinates[name].args[0])
        names = target.mesh.names
        key = names[axis] if axis < len(names) and names[axis] else str(axis)
        extent = flatten(target.mesh.layout).shape[axis]
        number = device.get(key, 0)
        if not isinstance(number, int) or isinstance(number, bool) or not 0 <= number < extent:
            raise ValueError(
                f"device coordinate {key!r} is {number!r}, and that mesh axis has "
                f"positions 0 to {extent - 1}"
            )
        placement = placement.intersect_params(isl.set(f"[{name}] -> {{ : {name} = {number} }}"))
        position = placement.find_dim_by_name(isl.dim_type.PARAM, name)
        placement = placement.project_out(isl.dim_type.PARAM, position, 1)
    return placement


def _placed(
    access: AccessRelation,
    local,
    logical,
    where: str,
    call,
    placement,
    bindings: IslParamValues,
) -> "isl.map":
    """One boundary's image carried from logical axes onto the positions it has.

    Held to the positions this participant was given, so which iterations are
    its own follows from the placement rather than from a relation that may
    reach past what it was handed. A coordinate past the logical value is cut
    before placing, since a regroup would fold it onto a position it does not
    name. A value with no placement is addressed at its own coordinates.
    """
    relation = access.relation
    if (
        not isinstance(local, TensorType)
        or not isinstance(logical, TensorType)
        or relation.is_empty()
    ):
        return relation
    if relation.dim(isl.dim_type.OUT) != len(logical.shape):
        raise ValueError(
            f"{type(call.target).__name__} reads {where} at "
            f"{relation.dim(isl.dim_type.OUT)} coordinates, and that value has "
            f"{len(logical.shape)} axes of its own; a canonical relation is "
            "stated in the axes an Op was written in"
        )
    placed_at = placement(logical)
    if placed_at is None:
        return _within_positions(relation, local)
    relation = relation.intersect_range(shape_to_isl_set(tuple(logical.shape), bindings))
    if placed_at.dim(isl.dim_type.OUT) != len(local.shape):
        raise ValueError(
            f"{type(call.target).__name__} places {where} at "
            f"{placed_at.dim(isl.dim_type.OUT)} positions, and one unit holds "
            f"{len(local.shape)}"
        )
    return _within_positions(relation.apply_range(placed_at), local)


def _within_positions(relation: "isl.map", local) -> "isl.map":
    """One relation held to the coordinates the value it reaches actually has."""
    if not isinstance(local, TensorType) or not is_literal_shape(local.shape):
        return relation
    box = shape_to_isl_set(tuple(local.shape), {})
    if box.tuple_dim() != relation.range().tuple_dim():
        return relation
    return relation.intersect_range(box)


def _addressed(
    relation: "isl.map", local, bindings: IslParamValues, where: str, call, *, lookup=False
) -> AccessRelation:
    """One placed boundary, held to the coordinates the value actually has.

    The projected relation is then the whole answer: what it reaches is what
    crossed, with nothing left for a reader to intersect again or to forget to.
    """
    held = _within_positions(relation, local)
    names = _parameter_names(held)
    return _held_countable(
        AccessRelation(held, {name: bindings[name] for name in names if name in bindings}, lookup),
        where,
        call,
    )


def _held_countable(access: AccessRelation, where: str, call) -> AccessRelation:
    """Refuse a projected boundary nobody can count.

    A relation is the only statement of how much crosses here, so one whose
    image is not a number leaves a reader with nothing -- and there is no
    falling back on what the Op said, because two answers is the thing this
    carrier exists to remove.
    """
    image = _reached_image(access)
    if image.dim(isl.dim_type.PARAM) or not image.is_bounded():
        raise ValueError(
            f"{type(call.target).__name__} states {where} as "
            f"{access.relation}, which reaches no countable number "
            "of elements here"
        )
    return access


def local_relations_of(call, ctx) -> tuple[AccessRelation, ...]:
    """One Op's relations, held against the Type in this reader's view.

    The Op stated its coordinates in its own axes; here they are carried onto the
    positions this reader addresses, and then held to the Call. What is checked
    is what needs the Type: one relation per result field after the arguments',
    each written at the rank that field has in this view.
    """
    return projected(relations_of(call, ctx), call, ctx)


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


def affine_term(value, name: str) -> "tuple[str, IslParamValues]":
    """One number of a relation, as a coefficient or as a bound parameter.

    A number written down is a coefficient of the map. Anything else is a
    parameter carrying the value it is, so whoever restricts the relation
    resolves it rather than reading its spelling. The caller states what it
    guarantees about the parameter; nothing is guaranteed here.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value), {}
    return name, {name: value}


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
    values: IslParamValues = {}
    for position in range(rank):
        begin, bound_begin = affine_term(offsets[position], f"o{position}")
        extent, bound_extent = affine_term(extents[position], f"e{position}")
        values.update(bound_begin)
        values.update(bound_extent)
        if bound_begin:
            guards.append(f"0 <= {begin}")
        if bound_extent:
            guards.append(f"1 <= {extent}")
            guards.extend(_at_most(extent, limits, position))
        if (bound_begin or bound_extent) and position < len(within):
            whole, bound_whole = affine_term(within[position], f"w{position}")
            values.update(bound_whole)
            guards.append(f"{begin} + {extent} <= {whole}")
        if begin == "0":
            guards.append(f"0 <= d{position} < {extent}")
            continue
        guards.append(f"{begin} <= d{position} < {begin} + {extent}")
    prefix = _declared(values)
    where = f" : {' and '.join(guards)}" if guards else ""
    reached = isl.map(f"{prefix}{{ [{domain}] -> [{domain}]{where} }}")
    whole = isl.map(f"{prefix}{{ [{domain}] -> [{domain}] }}")
    left = whole.subtract(reached).intersect_params(reached.params())
    return AccessRelation(left, values), AccessRelation(reached, values)


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
        elif static_dim_value(extent) == 1:
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
        if belongs[position] != axis or static_dim_value(extent) == 1:
            continue
        term = names[position] if stride == 1 else f"{stride} * {names[position]}"
        linear = term if not linear else f"{linear} + {term}"
        stride *= extent
    return linear or "0"


def iterating(
    extents: "Sequence", relations: tuple[AccessRelation, ...]
) -> tuple[AccessRelation, ...]:
    """Every boundary of one Op, on the iteration space that Op walks.

    An access map's domain is the Op's whole iteration space, so its bounds are
    the Op's and every boundary shares them -- not each boundary's own, and not
    a reader's guess from a result's rank. A contraction walks the axis it
    contracts; most Ops walk what they produce. A boundary may be partial in
    that space, which is one relation empty somewhere, not a second space.
    """
    relations = _by_identity(relations, len(tuple(extents)))
    values = _values_of(relations)
    try:
        domain = shape_to_isl_set(tuple(extents), values)
    except (TypeError, ValueError, isl.Error) as error:
        raise ValueError(
            f"an Op states it iterates {tuple(extents)}, which is no space to walk: {error}"
        ) from error
    return tuple(_held_to(access, domain, values) for access in relations)


def _by_identity(relations: tuple[AccessRelation, ...], rank: int) -> tuple[AccessRelation, ...]:
    """One Op's boundaries with one parameter name per value, and per value one name.

    A handler names each boundary's parameters on its own, so two boundaries can
    use one name for two values, or two names for one. Here the names are made
    the values': one object is one parameter across the Op, two objects are two,
    and a name a value already has is kept unless another value has it too or it
    names a coordinate -- of a boundary, or of the *rank*-dimensional space the
    Op is about to be held to. A value that cannot keep its name gets the first
    ``p<number>`` nothing here uses; the name carries no meaning of its own.
    """
    reserved = {f"d{index}" for index in range(rank)}
    owners: dict[str, set[int]] = {}
    for access in relations:
        for kind in (isl.dim_type.IN, isl.dim_type.OUT):
            for index in range(access.relation.dim(kind)):
                if access.relation.has_dim_name(kind, index):
                    reserved.add(access.relation.get_dim_name(kind, index))
        for name, value in access.values.items():
            owners.setdefault(name, set()).add(id(value))
    taken = set(owners) | reserved
    fresh = (f"p{number}" for number in itertools.count())
    canonical: dict[int, str] = {}
    for access in relations:
        for name, value in access.values.items():
            if id(value) in canonical:
                continue
            if len(owners[name]) > 1 or name in reserved:
                name = next(candidate for candidate in fresh if candidate not in taken)
                taken.add(name)
            canonical[id(value)] = name

    def renamed(access: AccessRelation) -> AccessRelation:
        targets = {name: canonical[id(value)] for name, value in access.values.items()}
        return AccessRelation(
            _renamed(access.relation, targets, taken),
            {canonical[id(value)]: value for value in access.values.values()},
            access.lookup,
        )

    return tuple(renamed(access) for access in relations)


def _renamed(relation: "isl.map", targets: dict[str, str], taken: set[str]) -> "isl.map":
    """*relation* with each parameter renamed to its target, two names for one merged.

    Every name that changes first moves to a name nobody uses, so swapping two
    names cannot capture either. A target already present is the same value, so
    the two parameters are equated and one is projected out.
    """
    staged: dict[str, str] = {}
    for name, target in targets.items():
        if name == target:
            continue
        temporary = f"__tf_rename_{len(staged)}"
        while temporary in taken:
            temporary = f"_{temporary}"
        position = relation.find_dim_by_name(isl.dim_type.PARAM, name)
        relation = relation.set_dim_name(isl.dim_type.PARAM, position, temporary)
        staged[temporary] = target
    for temporary, target in staged.items():
        position = relation.find_dim_by_name(isl.dim_type.PARAM, temporary)
        if relation.find_dim_by_name(isl.dim_type.PARAM, target) < 0:
            relation = relation.set_dim_name(isl.dim_type.PARAM, position, target)
            continue
        relation = relation.intersect_params(
            isl.set(f"[{temporary}, {target}] -> {{ : {temporary} = {target} }}")
        )
        position = relation.find_dim_by_name(isl.dim_type.PARAM, temporary)
        relation = relation.project_out(isl.dim_type.PARAM, position, 1)
    return relation


def _held_to(access: AccessRelation, domain: "isl.set", values: IslParamValues) -> AccessRelation:
    """One boundary, restricted to the coordinates its Op iterates."""
    relation = access.relation
    if relation.dim(isl.dim_type.IN) != domain.dim(isl.dim_type.SET):
        raise ValueError(
            f"a boundary is asked by {relation.dim(isl.dim_type.IN)} coordinates "
            f"and its Op iterates {domain.dim(isl.dim_type.SET)}; one Op states "
            "one coordinate system and every boundary of it answers about that one"
        )
    held = relation.intersect_domain(domain)
    names = _parameter_names(held)
    return AccessRelation(
        held, {name: values[name] for name in names if name in values}, access.lookup
    )


def projected_axes(access: AccessRelation) -> tuple[int | None, ...]:
    """Which one input axis, if any, each output axis projects from."""
    relation = access.relation
    source_rank = relation.dim(isl.dim_type.IN)
    axes = []
    for target_axis in range(relation.dim(isl.dim_type.OUT)):
        sources = []
        for source_axis in range(source_rank):
            local = isl.local_space.from_space(relation.get_space())
            equal = isl.constraint.alloc_equality(local)
            equal = equal.set_coefficient_si(isl.dim_type.IN, source_axis, 1)
            equal = equal.set_coefficient_si(isl.dim_type.OUT, target_axis, -1)
            projected = isl.map.universe(relation.get_space()).add_constraint(equal)
            if relation.is_subset(projected):
                sources.append(source_axis)
        axes.append(sources[0] if len(sources) == 1 else None)
    return tuple(axes)


def _as_number(value) -> int | None:
    """The number a bound parameter's value is, when it is one."""
    number = static_dim_value(value)
    if number is not None:
        return number
    inner = getattr(value, "value", None)
    return inner if isinstance(inner, int) and not isinstance(inner, bool) else None


def settled(access: AccessRelation) -> "isl.map":
    """One relation with every parameter fixed to a number.

    A parameter bound to something that has a value is fixed to that value. One
    whose value nobody here holds is fixed to the smallest the relation itself
    allows: the first legal iteration of a loop, the smallest legal window of a
    runtime extent. Either way a reader gets a number it can check rather than a
    range it has to interpret, so the parameters leave with their values put in.
    A relation nothing satisfies has no value to settle on.
    """
    relation = access.relation
    bound = access.values
    names = _parameter_names(relation)
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


def _reached_image(
    access: AccessRelation,
    box: "isl.set | None" = None,
    within: "isl.set | None" = None,
) -> "isl.set":
    """The distinct boundary coordinates reached in one occurrence."""
    relation = settled(access)
    if within is not None:
        relation = relation.intersect_domain(within)
    image = relation.range()
    if box is not None and box.tuple_dim() == image.tuple_dim():
        image = image.intersect(box)
    return image


def reached_elements(
    access: AccessRelation, box: "isl.set | None" = None, within: "isl.set | None" = None
) -> int | None:
    """How many distinct boundary elements one boundary reaches.

    Reaching the same element from many coordinates is one element moved, not
    many dependences, so an inner iteration axis costs nothing; and reaching
    past the coordinates the operand has is not reaching at all. This is one
    occurrence, so a parameter nobody bound settles at its first legal binding:
    how many crossings a loop performs is the footprint family's question.
    """
    image = _reached_image(access, box, within)
    if image.dim(isl.dim_type.PARAM):
        return None
    reached = cardinality(image)
    if not access.lookup or reached is None:
        return reached
    relation = settled(access)
    if within is not None:
        relation = relation.intersect_domain(within)
    if box is not None and box.tuple_dim() == relation.dim(isl.dim_type.OUT):
        relation = relation.intersect_range(box)
    coordinates = cardinality(relation.domain())
    return None if coordinates is None else min(reached, coordinates)


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


def reached_leaves(access: AccessRelation, count: int) -> "frozenset[int] | None":
    """Which of a structured value's flat leaves one boundary reaches.

    A tuple of numbers is indexed by one coordinate, so what crosses there is a
    set of leaves rather than a count: they need not be the same width, and
    charging the first for the one that was taken is a wrong number at the right
    size. Read at the same first legal binding one crossing is counted at.
    """
    image = settled(access).range()
    if image.dim(isl.dim_type.PARAM) or image.tuple_dim() != 1:
        return None
    return frozenset(
        leaf for leaf in range(count) if not image.intersect(isl.set(f"{{ [{leaf}] }}")).is_empty()
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


def control_read(rank: int, ctx, arg) -> AccessRelation:
    """The control numbers one operand carries, each read once.

    The domain is the result's positions like every other boundary, because a
    reader applies one execution domain to all of them and a boundary with a rank
    of its own is one it cannot answer. What it reaches is one point per number,
    so a reader counts the numbers rather than believing an empty set.
    """
    domain, image, reach = _control_space(rank, ctx, arg)
    where = f" : {reach}" if reach else ""
    return AccessRelation(isl.map(f"{{ [{domain}] -> [{image}]{where} }}"))


def reached_at(
    rank: int,
    local: "Type",
    logical: "Type",
    reads: dict,
    free: tuple = (),
) -> AccessRelation:
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
    values: IslParamValues = {}
    guards: list[str] = []
    for position, owner in enumerate(belongs):
        if owner not in free or local.shape[position] == 1:
            continue
        extent, bound = affine_term(local.shape[position], f"n{position}")
        values.update(bound)
        image[position] = f"g{position}"
        guards.append(f"0 <= g{position} < {extent}")
    domain = ", ".join(f"d{index}" for index in range(rank))
    where = f" : {' and '.join(guards)}" if guards else ""
    return AccessRelation(
        isl.map(f"{_declared(values)}{{ [{domain}] -> [{', '.join(image)}]{where} }}"),
        values,
        lookup=bool(free),
    )


def window_source(
    offsets: tuple,
    rank: int,
    local: "Type",
    logical: "Type",
    carried: dict,
    extents: tuple = (),
    limits: tuple = (),
) -> AccessRelation:
    """One operand read at its own coordinates, from where a window put them.

    A window covers the operand's shape wherever it lands, so the coordinate read
    is the one reached shifted back to where the window starts. The shift is per
    logical axis, because that is what an offset is stated against, and only then
    spread over the positions this operand's own layout made -- which are not the
    result's. An axis whose extent is given holds the read to it, because an
    operand supplying more than the window takes is not read past it; an axis
    given ``None`` is covered whole and asks for no such guard.
    """
    values: IslParamValues = {}
    reads: list[str] = []
    guards: list[str] = []
    for axis in range(len(logical.shape)):
        walked = carried.get(axis, "0")
        begin, bound = affine_term(offsets[axis] if axis < len(offsets) else 0, f"o{axis}")
        values.update(bound)
        if bound:
            guards.append(f"0 <= {begin}")
        reads.append(walked if begin == "0" else f"{walked} - {begin}")
        if axis >= len(extents) or extents[axis] is None:
            continue
        extent, bound_extent = affine_term(extents[axis], f"e{axis}")
        values.update(bound_extent)
        if bound_extent:
            guards.append(f"1 <= {extent}")
            guards.extend(_at_most(extent, limits, axis))
        guards.append(
            f"0 <= {walked} - {begin} < {extent}" if begin != "0" else f"0 <= {walked} < {extent}"
        )
    domain = ", ".join(f"d{index}" for index in range(rank))
    image = ", ".join(factored_image(reads, local, logical))
    where = f" : {' and '.join(guards)}" if guards else ""
    return AccessRelation(
        isl.map(f"{_declared(values)}{{ [{domain}] -> [{image}]{where} }}"), values
    )


def _declared(values: IslParamValues) -> str:
    """The parameter list a relation built from *values* declares, or nothing."""
    return f"[{', '.join(values)}] -> " if values else ""


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


def identity_access(rank: int) -> AccessRelation:
    """Identity boundary for a tensor of *rank*: each element read where it is."""
    dims = ", ".join(f"d{index}" for index in range(rank))
    return AccessRelation(isl.map(f"{{ [{dims}] -> [{dims}] }}" if rank else "{ [] -> [] }"))


def is_one(expr) -> bool:
    """Return True for any shape entry that represents the literal 1.

    Shape entries can be either ``Constant(value=1)`` (the canonical IR
    form, produced by the parser / annotation lift) or a Python ``int``
    (produced ad-hoc by some typeinfer rules — Reduce, Slice, etc.). Both
    forms must broadcast against larger dims; restricting to ``Constant``
    only breaks the ``(1, N) ⊕ (1, 1)`` pattern that falls out of
    ``Reduce(..., keepdim=True)``.
    """
    if isinstance(expr, Constant) and expr.value == 1:
        return True
    if isinstance(expr, int) and not isinstance(expr, bool) and expr == 1:
        return True
    return False


def broadcast_shapes(a: tuple, b: tuple, *, raising: bool = True):
    """NumPy-style right-aligned broadcast on ``tuple[Expr, ...]``.

    The shorter shape is padded on the left with 1s, then dims combine
    pairwise (equal, or one is 1). For an incompatible pair: raises
    ``ValueError`` when *raising* (the default), else returns ``None``.
    """
    if a == b:
        return a
    n = max(len(a), len(b))
    ap = (1,) * (n - len(a)) + tuple(a)
    bp = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(ap, bp):
        if x == y:
            out.append(x)
        elif is_one(x):
            out.append(y)
        elif is_one(y):
            out.append(x)
        elif raising:
            raise ValueError(f"cannot broadcast shapes {a} and {b}")
        else:
            return None
    return tuple(out)


def broadcast_access(result_shape: tuple, operand_shape: tuple) -> AccessRelation:
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
        "0" if operand_shape[index - offset] == 1 else dims[index] for index in range(offset, rank)
    ]
    domain = ", ".join(dims)
    if not reads:
        return AccessRelation(isl.map(f"{{ [{domain}] -> [] }}" if rank else "{ [] -> [] }"))
    return AccessRelation(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def _operand_reads(
    shape: tuple,
    out_shape: tuple,
    out_axes: tuple[str, ...],
    inner: str,
    *,
    kept_axis: int,
    output_axis: int,
    contraction_axis: int,
) -> list[str]:
    """One coordinate per axis of an operand of the contraction.

    The two matrix axes are the contraction and the one the result keeps; which
    is which is the only difference between the two operands. Batch axes are
    right-aligned against the result's, and one the operand broadcasts reads its
    only coordinate rather than the result's.
    """
    rank = len(shape)
    kept_axis %= rank
    contraction_axis %= rank
    shift = len(out_shape) - rank
    reads: list[str] = []
    for axis in range(rank):
        if axis == contraction_axis:
            reads.append(inner)
        elif axis == kept_axis:
            reads.append(out_axes[output_axis])
        elif is_one(shape[axis]) and not is_one(out_shape[axis + shift]):
            reads.append("0")
        else:
            reads.append(out_axes[axis + shift])
    return reads


def matmul_relations(
    lhs_shape: tuple,
    rhs_shape: tuple,
    axes: tuple[int, int, int, int],
) -> tuple[AccessRelation, ...]:
    """Every coordinate of each operand a contraction reaches, read once.

    A product walks the axis it sums, so that axis is a coordinate this Op is
    asked by rather than something existential inside an image, and the result is
    accumulated over it. Reading an operand at the result's own coordinates would
    claim a shape it does not have the moment the summed and kept axes differ in
    extent. Which positions any of these coordinates are is the reader's
    question; the result's own extents follow from the operands.
    """
    a_m, a_k, b_n, b_k = axes
    batch = broadcast_shapes(tuple(lhs_shape[:-2]), tuple(rhs_shape[:-2]), raising=False)
    if batch is None:
        raise ValueError(
            f"MatMul batches {tuple(lhs_shape[:-2])} against "
            f"{tuple(rhs_shape[:-2])}, which do not broadcast"
        )
    out_shape = (*batch, lhs_shape[a_m], rhs_shape[b_n])
    summed = lhs_shape[a_k]
    out_axes = (*(f"d{index}" for index in range(len(batch))), "m", "n")
    dims = ", ".join((*out_axes, "k"))
    inner = "0" if is_one(summed) else "k"
    inputs = []
    for shape, kept_axis, output_axis, contraction_axis in (
        (tuple(lhs_shape), a_m, -2, a_k),
        (tuple(rhs_shape), b_n, -1, b_k),
    ):
        reads = _operand_reads(
            shape,
            out_shape,
            out_axes,
            inner,
            kept_axis=kept_axis,
            output_axis=output_axis,
            contraction_axis=contraction_axis,
        )
        inputs.append(AccessRelation(isl.map(f"{{ [{dims}] -> [{', '.join(reads)}] }}")))
    accumulates = ", ".join(out_axes)
    return iterating(
        (*out_shape, summed),
        (*inputs, AccessRelation(isl.map(f"{{ [{dims}] -> [{accumulates}] }}"))),
    )


def unread_access(domain_rank: int, rank: int) -> AccessRelation:
    """No coordinate of a rank-*rank* value is reached from a rank-*domain_rank* space."""
    domain = ", ".join(f"d{index}" for index in range(domain_rank))
    reads = ", ".join(f"i{index}" for index in range(rank))
    return AccessRelation(isl.map(f"{{ [{domain}] -> [{reads}] : 1 = 0 }}"))


def readnone_relations(
    result_shape: Callable[[tuple], tuple],
) -> Callable[..., tuple[AccessRelation, ...]]:
    """Handler for an Op that answers from its operands' Types, not their elements.

    A rank, a shape, a view of the same value: the answer is in the Types, so no
    coordinate is read and nothing crosses. *result_shape* derives the result's
    shape from the operand Types alone, because type inference asks this before
    the Call has a Type of its own.
    """

    def _handler(call, ctx) -> tuple[AccessRelation, ...]:
        types = tuple(ctx.type_of(arg) for arg in call.args)
        shape = tuple(result_shape(types))
        return iterating(
            shape,
            (
                *(unread_access(len(shape), len(getattr(type_, "shape", ()))) for type_ in types),
                unread_access(len(shape), len(shape)),
            ),
        )

    return _handler


def linearized_view(out_shape: tuple, in_shape: tuple) -> AccessRelation:
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
        raise ValueError(f"a view relabels a shape it can count: {out_shape!r} from {in_shape!r}")
    out_rank, in_rank = len(out_shape), len(in_shape)
    if 0 in out_shape or 0 in in_shape:
        dims = ", ".join(f"d{index}" for index in range(out_rank))
        reads = ", ".join("0" for _ in range(in_rank))
        return AccessRelation(isl.map(f"{{ [{dims}] -> [{reads}] : 1 = 0 }}"))
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
    return AccessRelation(isl.map(f"{{ [{domain}] -> [{', '.join(reads)}] }}"))


def identity_relations(call, ctx) -> tuple[AccessRelation, ...]:
    """Walk the operands' broadcast domain and map each boundary into it.

    Structural operands whose shape is not inferred yet borrow the domain.
    Only operand types are read: HIR type inference derives its result here.
    Incompatible shapes expose an invalid relation instead of guessing a domain.
    """
    types = tuple(ctx.type_of(arg) for arg in call.args)
    shapes = tuple(tuple(ty.shape) if hasattr(ty, "shape") else None for ty in types)
    domain = reduce(broadcast_shapes, (shape for shape in shapes if shape is not None), ())

    def access(shape):
        return identity_access(len(domain)) if shape is None else broadcast_access(domain, shape)

    return iterating(domain, (*(access(shape) for shape in shapes), identity_access(len(domain))))


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
    "AccessRelation",
    "access_relation_registry",
    "iterating",
    "identity_access",
    "identity_relations",
    "logical_axes_of",
    "logical_coordinates",
    "local_relations_of",
    "placed_window",
    "projected",
    "projected_axes",
    "readnone_relations",
    "leaves_of",
    "reached_elements",
    "reached_leaves",
    "register_access_relation",
    "relations_of",
    "renaming_relation",
    "settled",
    "shape_from_relation",
    "static_bytes",
    "unread_access",
    "window_source",
]
