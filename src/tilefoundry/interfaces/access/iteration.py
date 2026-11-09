"""Put every boundary of one Op on the iteration space that Op walks."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import isl

from tilefoundry import utils
from tilefoundry.ir import dim

from .registry import iteration_universe, values_of
from .relation import Relation, parameter_names


def iterating(extents: "Sequence", relations: tuple[Relation, ...]) -> tuple[Relation, ...]:
    """Every boundary of one Op, on the iteration space that Op walks.

    An access map's domain is the Op's whole iteration space, so its bounds are
    the Op's and every boundary shares them -- not each boundary's own, and not
    a reader's guess from a result's rank. A contraction walks the axis it
    contracts; most Ops walk what they produce. A boundary may be partial in
    that space, which is one relation empty somewhere, not a second space.
    """
    relations = by_identity(relations, len(tuple(extents)))
    values = values_of(relations)
    try:
        domain = dim.shape_to_isl_set(tuple(extents), values)
    except (TypeError, ValueError, isl.Error) as error:
        raise ValueError(
            f"an Op states it iterates {tuple(extents)}, which is no space to walk: {error}"
        ) from error
    return tuple(held_to(access, domain, values) for access in relations)


def by_identity(relations: tuple[Relation, ...], rank: int) -> tuple[Relation, ...]:
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

    def renamed(access: Relation) -> Relation:
        targets = {name: canonical[id(value)] for name, value in access.values.items()}
        return Relation(
            rename_parameters(access.relation, targets, taken),
            {canonical[id(value)]: value for value in access.values.values()},
            access.may_access,
        )

    return tuple(renamed(access) for access in relations)


def rename_parameters(relation: "isl.map", targets: dict[str, str], taken: set[str]) -> "isl.map":
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


def held_to(access: Relation, domain: "isl.set", values: dim.IslParamValues) -> Relation:
    """One boundary, restricted to the coordinates its Op iterates."""
    relation = access.relation
    if relation.dim(isl.dim_type.IN) != domain.dim(isl.dim_type.SET):
        raise ValueError(
            f"a boundary is asked by {relation.dim(isl.dim_type.IN)} coordinates "
            f"and its Op iterates {domain.dim(isl.dim_type.SET)}; one Op states "
            "one coordinate system and every boundary of it answers about that one"
        )
    try:
        affine = utils.as_multi_aff(relation) if relation.is_single_valued() else None
    except ValueError:
        affine = None
    held = relation.intersect_domain(domain)
    names = parameter_names(held)
    return Relation(
        held, {name: values[name] for name in names if name in values}, access.may_access, affine
    )


def iteration_geometry(relations) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Static extents and authored axis names of one relation domain."""
    domain = iteration_universe(relations)
    if domain is None:
        raise ValueError("access relations state no iteration space")
    extents, names = [], []
    for axis in range(domain.tuple_dim()):
        low, high = domain.dim_min_val(axis), domain.dim_max_val(axis)
        if not (low.is_int() and high.is_int() and low.get_num_si() == 0):
            raise ValueError(f"iteration axis {axis} is not a zero-based static extent")
        extents.append(high.get_num_si() + 1)
        names.append(domain.get_dim_name(isl.dim_type.SET, axis) or str(axis))
    return tuple(extents), tuple(names)


__all__ = [
    "iterating",
    "iteration_geometry",
]
