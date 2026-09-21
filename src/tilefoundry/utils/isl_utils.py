"""Set and map operations shared by everything that reads an isl relation."""

from __future__ import annotations

from itertools import product

import isl

__all__ = ["as_multi_aff", "cardinality", "equates", "involved_dims"]


def _count(image: "isl.set") -> int | None:
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


def _param_corners(image: "isl.set") -> tuple["isl.set", ...] | None:
    """Fix every parameter to each feasible corner of its bounded box."""
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
    values: list[tuple[int, ...]] = []
    for axis in range(count):
        if not axes.dim_is_bounded(isl.dim_type.SET, axis):
            return None
        low = axes.dim_min_val(axis)
        high = axes.dim_max_val(axis)
        if not (low.is_int() and high.is_int()):
            return None
        lo, hi = low.get_num_si(), high.get_num_si()
        values.append((lo,) if lo == hi else (lo, hi))
    corners = []
    for point in product(*values):
        corner = context
        for axis, value in enumerate(point):
            corner = corner.fix_si(isl.dim_type.PARAM, axis, value)
        if not corner.is_empty():
            corners.append(corner)
    return tuple(corners)


def cardinality(image: "isl.set") -> int | None:
    """Return a finite point count, or a corner-wise upper bound with params.

    A box is counted from its bounds: every axis is an independent bounded
    interval, so their lengths multiply, at a cost set by rank alone. ISL's own
    count has a closed form for simple sets and loses it as rank grows: on one
    462M-point image, 4.0 ms at two dimensions and 2.3 s at four. The cost
    tracks rank, not points. Anything that is not a box falls back to it. When
    bounded parameters remain free, each feasible box corner is counted and the
    maximum is returned; an unbounded parameter has no finite answer.
    """
    corners = _param_corners(image)
    if corners is None:
        return None
    counts = tuple(_count(image.intersect_params(corner)) for corner in corners)
    return None if not counts or any(count is None for count in counts) else max(counts)


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
