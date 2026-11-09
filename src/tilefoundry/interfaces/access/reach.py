"""What a boundary reaches in one occurrence: elements, leaves and bytes."""

from __future__ import annotations

import isl

from tilefoundry import utils
from tilefoundry.ir import expr, types
from tilefoundry.ir import utils as ir_utils

from .relation import Relation, parameter_names


def settled(access: Relation) -> "isl.map":
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
    names = parameter_names(relation)
    if not names:
        return relation
    if relation.is_empty():
        return relation.project_out(isl.dim_type.PARAM, 0, len(names))
    space = f"[{', '.join(names)}] -> "
    legal = relation.params()
    for name in names:
        number = as_number(bound.get(name))
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


def as_number(value) -> int | None:
    """The number a bound parameter's value is, when it is one."""
    number = expr.static_dim_value(value)
    if number is not None:
        return number
    inner = getattr(value, "value", None)
    return inner if isinstance(inner, int) and not isinstance(inner, bool) else None


def reached_image(
    access: Relation,
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
    access: Relation, box: "isl.set | None" = None, within: "isl.set | None" = None
) -> int | None:
    """How many distinct boundary elements one boundary reaches.

    Reaching the same element from many coordinates is one element moved, not
    many dependences, so an inner iteration axis costs nothing; and reaching
    past the coordinates the operand has is not reaching at all. This is one
    occurrence, so a parameter nobody bound settles at its first legal binding:
    how many crossings a loop performs is the footprint family's question.
    """
    image = reached_image(access, box, within)
    if image.dim(isl.dim_type.PARAM):
        return None
    reached = utils.cardinality(image)
    if not access.may_access or reached is None:
        return reached
    relation = settled(access)
    if within is not None:
        relation = relation.intersect_domain(within)
    if box is not None and box.tuple_dim() == relation.dim(isl.dim_type.OUT):
        relation = relation.intersect_range(box)
    coordinates = utils.cardinality(relation.domain())
    return reached if coordinates is None else min(reached, coordinates)


def reached_leaves(access: Relation, count: int) -> "frozenset[int] | None":
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


def leaf_paths(type_: "types.Type", path: tuple[int, ...] = ()):
    """Yield each nested leaf's tuple path and type, including non-tensor leaves."""
    if isinstance(type_, types.TupleType):
        for index, field in enumerate(type_.fields):
            yield from leaf_paths(field, (*path, index))
    else:
        yield path, type_


def static_bytes(type_: "types.Type") -> int | None:
    """How many bytes a Type holds, or ``None`` when it is not static."""
    if isinstance(type_, types.TupleType):
        sizes = [static_bytes(field_) for field_ in type_.fields]
        if any(size is None for size in sizes):
            return None
        return sum(size for size in sizes if size is not None)
    if not isinstance(type_, types.TensorType):
        return None
    if not all(
        isinstance(dim_value, int) and not isinstance(dim_value, bool) for dim_value in type_.shape
    ):
        return None
    try:
        amount = ir_utils.tensor_bytes(type_)
    except (TypeError, ValueError):
        return None
    return amount if isinstance(amount, int) else None


__all__ = [
    "leaf_paths",
    "reached_elements",
    "reached_image",
    "reached_leaves",
    "settled",
    "static_bytes",
]
