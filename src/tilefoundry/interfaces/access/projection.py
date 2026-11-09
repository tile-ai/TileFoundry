"""Project a boundary into the coordinates one reader can address."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isl

from tilefoundry.ir import dim, expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple

from .alias import aliased_operand
from .reach import reached_image
from .registry import iteration_universe, relations_of, values_of
from .relation import Relation, parameter_names


def projected(
    relations: tuple[Relation, ...],
    call,
    ctx,
    *,
    device: "Mapping[str, int] | None" = None,
) -> tuple[Relation, ...]:
    """Every boundary in the coordinates the reader asking can address.

    Compose logical accesses with layout positions, restricting participants
    and boundaries to shared iterations. Without a topology level coordinates
    stay logical; otherwise *device* supplies each mesh coordinate, keyed by
    axis name (or number when unnamed), defaulting to zero. Inputs come first.
    Unplaced, parameter-free boundaries own every iteration they answer about,
    so their unchanged logical view needs no cross-boundary restriction.
    """
    ninputs = len(call.args)
    held = ctx.local_type_of(call)
    fields = held.fields if isinstance(held, types.TupleType) else (held,)
    logical = ctx.type_of(call)
    logical_fields = logical.fields if isinstance(logical, types.TupleType) else (logical,)
    if len(relations) - ninputs != len(fields):
        raise ValueError(
            f"{type(call.target).__name__} describes {len(relations) - ninputs} output "
            f"boundar{'y' if len(relations) - ninputs == 1 else 'ies'} of a call "
            f"with {len(fields)}"
        )
    bindings = values_of(relations)
    level = getattr(ctx, "topology_level", None)
    coordinates: dim.IslParamValues = {}

    def placement(value) -> "isl.map | None":
        shard = types.shard_layout_of(getattr(value, "layout", None))
        if level is None or not isinstance(value, types.TensorType) or shard is None:
            return None
        placed_at = dim.layout_to_isl_map(
            tuple(value.shape),
            value.layout,
            coordinates,
            divided=lambda layer: ir_utils.divided_mesh_axes(
                layer, topology_level=level, topologies=ctx.topologies
            ),
        )
        return at_device(placed_at, coordinates, device or {})

    views = (
        *((ctx.local_type_of(arg), ctx.type_of(arg)) for arg in call.args),
        *zip(fields, logical_fields, strict=True),
    )
    where = tuple(boundary(index, ninputs) for index in range(len(relations)))
    placed = tuple(
        place_boundary(access, *view, label, call, placement, bindings)
        for access, view, label in zip(relations, views, where, strict=True)
    )
    carried = tuple(
        address(relation, bindings, label, call, may_access=access.may_access)
        for relation, label, access in zip(placed, where, relations, strict=True)
    )
    if (
        level is None
        and all(local is logical for local, logical in views)
        and all(not access.values for access in relations)
        and all(
            access.relation.get_space()
            .domain()
            .is_equal(relations[0].relation.get_space().domain())
            for access in relations[1:]
        )
    ):
        return carried
    answered = tuple(
        answered_domain(access, view[1]) for access, view in zip(relations, views, strict=True)
    )
    share = own_iterations(relations, placed, answered, ninputs)
    if share is None:
        return carried
    return tuple(hold_to_share(access, share, bindings) for access in carried)


def answered_domain(access: Relation, logical) -> "isl.set":
    """Where a boundary answers about coordinates the value actually has.

    A relation may be written to reach past its value -- a window shifted back
    to where it came from does -- and outside that it is saying nothing rather
    than saying this participant does not iterate there.
    """
    relation = access.relation
    if not isinstance(logical, types.TensorType) or not expr.is_literal_shape(logical.shape):
        return relation.domain()
    box = dim.shape_to_isl_set(tuple(logical.shape), {})
    if box.tuple_dim() != relation.range().tuple_dim():
        return relation.domain()
    return relation.intersect_range(box).domain()


def own_iterations(
    stated: tuple[Relation, ...], placed: tuple, answered: tuple, ninputs: int
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
                f"{boundary(index, ninputs)} cannot be lined up with the space its "
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


def hold_to_share(access: Relation, share: "isl.set", bindings: dim.IslParamValues) -> Relation:
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
    names = parameter_names(held)
    return Relation(
        held, {name: bindings[name] for name in names if name in bindings}, access.may_access
    )


def renaming_relation(call, ctx, local_relations: tuple[Relation, ...]) -> Relation:
    """One view's own coordinates, as coordinates of the value it renames.

    A view states where it reads and where it writes over one space, so going
    from its result's coordinates to its source's is reading the second
    backwards and the first forwards. Every consumer that folds a view into its
    buffer asks for this rather than rebuilding the Op's arithmetic, which is
    how one relation answers dependence, footprint and movement alike.
    ``local_relations`` is the call's relations projected into ``ctx``'s window.
    """
    if isinstance(ctx.type_of(call.args[0]), types.TupleType):
        raise ValueError(
            f"{type(call.target).__name__} renames a field of a tuple, which is "
            "one leaf of it rather than a coordinate change to fold"
        )
    written = local_relations[len(call.args)].relation
    reads = local_relations[aliased_operand(call)].relation
    folded = written.reverse().apply_range(reads)
    bindings = values_of(local_relations)
    names = parameter_names(folded)
    return Relation(
        folded,
        {name: bindings[name] for name in names if name in bindings},
    )


def shape_from_relation(access: Relation, extents: "Sequence") -> tuple:
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
        dim.isl_to_dimension(image.dim_max(axis).add_constant(1), access.values)
        for axis in range(rank)
    )


def at_device(placement: "isl.map", coordinates: dim.IslParamValues, device: Mapping) -> "isl.map":
    """*placement* at one unit: each mesh coordinate fixed, then gone.

    A coordinate is fixed to the value *device* gives under its mesh axis's
    name, or 0. A value that is not one of that axis's positions is refused,
    rather than fixing the unit nowhere and counting nothing.
    """
    for name in parameter_names(placement):
        target = coordinates[name].target
        axis = expr.static_dim_value(coordinates[name].args[0])
        names = target.mesh.names
        key = names[axis] if axis < len(names) and names[axis] else str(axis)
        extent = types.flatten(target.mesh.layout).shape[axis]
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


def place_boundary(
    access: Relation,
    local,
    logical,
    where: str,
    call,
    placement,
    bindings: dim.IslParamValues,
) -> "isl.map":
    """One boundary's image carried from logical axes onto the positions it has.

    Held to the positions this participant was given, so which iterations are
    its own follows from the placement rather than from a relation that may
    reach past what it was handed. A coordinate past the logical value is cut
    before placing, since a regroup would fold it onto a position it does not
    name. A value with no placement is addressed at its own coordinates.
    """
    relation = access.relation
    if not isinstance(local, types.TensorType) or not isinstance(logical, types.TensorType):
        return within_positions(relation, local)
    if relation.dim(isl.dim_type.OUT) != len(logical.shape):
        raise ValueError(
            f"{type(call.target).__name__} reads {where} at "
            f"{relation.dim(isl.dim_type.OUT)} coordinates, and that value has "
            f"{len(logical.shape)} axes of its own; a canonical relation is "
            "stated in the axes an Op was written in"
        )
    placed_at = placement(logical)
    if placed_at is None:
        return within_positions(relation, local)
    relation = relation.intersect_range(dim.shape_to_isl_set(tuple(logical.shape), bindings))
    if placed_at.dim(isl.dim_type.OUT) != len(local.shape):
        raise ValueError(
            f"{type(call.target).__name__} places {where} at "
            f"{placed_at.dim(isl.dim_type.OUT)} positions, and one unit holds "
            f"{len(local.shape)}"
        )
    return within_positions(relation.apply_range(placed_at), local)


def within_positions(relation: "isl.map", local) -> "isl.map":
    """One relation held to the coordinates the value it reaches actually has."""
    if not isinstance(local, types.TensorType) or not expr.is_literal_shape(local.shape):
        return relation
    box = dim.shape_to_isl_set(tuple(local.shape), {})
    if box.tuple_dim() != relation.range().tuple_dim():
        return relation
    return relation.intersect_range(box)


def address(
    relation: "isl.map", bindings: dim.IslParamValues, where: str, call, *, may_access=False
) -> Relation:
    """One placed boundary, held to the coordinates the value actually has.

    The projected relation is then the whole answer: what it reaches is what
    crossed, with nothing left for a reader to intersect again or to forget to.
    """
    names = parameter_names(relation)
    return held_countable(
        Relation(
            relation, {name: bindings[name] for name in names if name in bindings}, may_access
        ),
        where,
        call,
    )


def held_countable(access: Relation, where: str, call) -> Relation:
    """Refuse a projected boundary nobody can count.

    A relation is the only statement of how much crosses here, so one whose
    image is not a number leaves a reader with nothing -- and there is no
    falling back on what the Op said, because two answers is the thing this
    carrier exists to remove.
    """
    image = reached_image(access)
    if image.dim(isl.dim_type.PARAM) or not image.is_bounded():
        raise ValueError(
            f"{type(call.target).__name__} states {where} as "
            f"{access.relation}, which reaches no countable number "
            "of elements here"
        )
    return access


def local_relations_of(call, ctx) -> tuple[Relation, ...]:
    """One Op's relations, held against the Type in this reader's view.

    The Op stated its coordinates in its own axes; here they are carried onto the
    positions this reader addresses, and then held to the Call. What is checked
    is what needs the Type: one relation per result field after the arguments',
    each written at the rank that field has in this view.
    """
    return projected(relations_of(call, ctx), call, ctx)


def logical_axes_of(local: "types.Type", logical: "types.Type") -> list[int]:
    """Which logical axis each axis of a projected Type belongs to.

    A canonical `ShardLayout` factors a logical axis into several layout
    positions -- an extent of 12 over a mesh of 6 becomes `(6, 2)` -- and the
    projected Type keeps those positions. So the projected rank is not the
    authored rank and an Op's own axis numbers do not index it: reducing "axis
    1" by position would reduce the mesh factor of axis 0 and carry its residual
    through. A plain projected view already names logical axes even when its
    engine retains a larger source domain, as Local does.
    """
    layout = getattr(local, "layout", None)
    if types.shard_layout_of(layout) is None and len(local.shape) == len(logical.shape):
        return int_tuple.mode_axes(
            int_tuple.compatible_profile(tuple(logical.shape), tuple(logical.shape))
        )
    inner = getattr(layout, "layout", None)
    shape = getattr(inner, "shape", None) or getattr(layout, "shape", None)
    if shape is None or len(shape) != len(local.shape):
        return list(range(len(local.shape)))
    return int_tuple.mode_axes(int_tuple.compatible_profile(tuple(shape), tuple(logical.shape)))


def logical_coordinates(local: "types.Type", logical: "types.Type") -> dict[int, str]:
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


def affine_term(value, name: str) -> "tuple[str, dim.IslParamValues]":
    """One number of a relation, as a coefficient or as a bound parameter.

    A number written down is a coefficient of the map. Anything else is a
    parameter carrying the value it is, so whoever restricts the relation
    resolves it rather than reading its spelling. The caller states what it
    guarantees about the parameter; nothing is guaranteed here.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value), {}
    return name, {name: value}


def projected_axes(access: Relation) -> tuple[int | None, ...]:
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


def logical_term(
    names: "Sequence[str]", local: "types.Type", logical: "types.Type", axis: int
) -> str:
    """One logical axis's coordinate, rebuilt from the positions holding it."""
    linear, stride = "", 1
    belongs = logical_axes_of(local, logical)
    for position in reversed(range(len(belongs))):
        extent = local.shape[position]
        if belongs[position] != axis or expr.static_dim_value(extent) == 1:
            continue
        term = names[position] if stride == 1 else f"{stride} * {names[position]}"
        linear = term if not linear else f"{linear} + {term}"
        stride *= extent
    return linear or "0"


def factored_image(reads: "Sequence[str]", local: "types.Type", logical: "types.Type") -> list[str]:
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


def boundary(index: int, ninputs: int) -> str:
    """Where one relation of an Op's tuple is, for a message."""
    return f"input {index}" if index < ninputs else f"output {index - ninputs}"


__all__ = [
    "affine_term",
    "factored_image",
    "local_relations_of",
    "logical_axes_of",
    "logical_coordinates",
    "logical_term",
    "projected",
    "projected_axes",
    "renaming_relation",
    "shape_from_relation",
]
