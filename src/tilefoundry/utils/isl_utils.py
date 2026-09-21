"""Set and map operations shared by everything that reads an isl relation."""

from __future__ import annotations

from itertools import product

import isl

__all__ = [
    "as_multi_aff",
    "cardinality",
    "count",
    "equates",
    "involved_dims",
    "param_points",
]

_PARAM_POINT_LIMIT = 4096


def count(image: "isl.set") -> int | None:
    """Count *image* after every parameter has been fixed by its caller."""
    image = image.coalesce()
    if image.is_box():
        amount = 1
        for axis in range(image.tuple_dim()):
            if not image.dim_is_bounded(isl.dim_type.SET, axis):
                break
            low, high = image.dim_min_val(axis), image.dim_max_val(axis)
            if not (low.is_int() and high.is_int()):
                break
            amount *= high.get_num_si() - low.get_num_si() + 1
        else:
            return amount
    amount = image.count_val()
    return amount.get_num_si() if amount.is_int() else None


def param_points(image: "isl.set") -> tuple["isl.set", ...] | None:
    """Fix parameters to every feasible point of a small bounded integer box."""
    context = image.params()
    count = context.dim(isl.dim_type.PARAM)
    if not count:
        return (context,)
    axes = context.move_dims(
        isl.dim_type.SET,
        0,
        isl.dim_type.PARAM,
        0,
        count,
    )
    values: list[range] = []
    box_points = 1
    for axis in range(count):
        if not axes.dim_is_bounded(isl.dim_type.SET, axis):
            return None
        low = axes.dim_min_val(axis)
        high = axes.dim_max_val(axis)
        if not (low.is_int() and high.is_int()):
            return None
        lo, hi = low.get_num_si(), high.get_num_si()
        choices = range(lo, hi + 1)
        box_points *= len(choices)
        if box_points > _PARAM_POINT_LIMIT:
            return None
        values.append(choices)
    points = []
    for point in product(*values):
        fixed = context
        for axis, value in enumerate(point):
            fixed = fixed.fix_si(isl.dim_type.PARAM, axis, value)
        if not fixed.is_empty():
            points.append(fixed)
    return tuple(points)


def cardinality(image: "isl.set") -> int | None:
    """Return a finite point count, maximizing over a small parameter box.

    A box is counted by multiplying its bounded axis lengths, at a cost set by
    rank alone; anything else falls back to ISL's count. With bounded free
    parameters, every feasible integer point in a box of at most 4096 points is
    counted and the true maximum is returned. A larger or unbounded parameter
    box has no answer.
    """
    if not image.dim(isl.dim_type.PARAM):
        return count(image)
    points = param_points(image)
    if points is None:
        return None
    counts = tuple(count(image.intersect_params(point)) for point in points)
    if any(amount is None for amount in counts):
        return None
    return max(counts, default=0)


def equates(relation: "isl.map", out_axis: int, in_dim: int) -> bool:
    """Whether *relation* everywhere sends domain dim *in_dim* to *out_axis*.

    Asked of a space, not of one pair: the equality is built over the whole
    space and *relation* is tested against it, so an axis that merely happens to
    hold that value somewhere does not count.
    """
    params = [
        relation.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(relation.dim(isl.dim_type.PARAM))
    ]
    prefix = f"[{', '.join(params)}] -> " if params else ""
    reads = ", ".join(f"i{index}" for index in range(relation.dim(isl.dim_type.IN)))
    writes = ", ".join(f"o{index}" for index in range(relation.dim(isl.dim_type.OUT)))
    equality = isl.map(f"{prefix}{{ [{reads}] -> [{writes}] : o{out_axis} = i{in_dim} }}")
    return bool(relation.is_subset(equality))


def as_multi_aff(relation: "isl.map") -> "isl.multi_aff":
    """The one affine access *relation* is, even where its domain is restricted.

    Raises ValueError when it is piecewise, because a caller asking for one
    access has no reading for two.
    """
    pieces: list[isl.multi_aff] = []
    relation.as_pw_multi_aff().foreach_piece(lambda _domain, access: pieces.append(access))
    if len(pieces) != 1:
        raise ValueError(f"expected one affine access piece, got {len(pieces)}")
    return pieces[0]


def involved_dims(relation: "isl.map") -> "set[int]":
    """Every domain dim any result axis of *relation* reads.

    Includes dims that appear only inside an access that is not a projection.
    """
    access = as_multi_aff(relation)
    dims: set[int] = set()
    for out_axis in range(access.dim(isl.dim_type.OUT)):
        affine = access.get_at(out_axis)
        for in_dim in range(access.dim(isl.dim_type.IN)):
            if int(affine.get_coefficient_val(isl.dim_type.IN, in_dim).num_si()) != 0:
                dims.add(in_dim)
    return dims
