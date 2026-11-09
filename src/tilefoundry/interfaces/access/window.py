"""Boundaries of an Op that places a window over a value."""

from __future__ import annotations

import isl

from tilefoundry.ir import dim, expr, types
from tilefoundry.ir import utils as ir_utils

from .projection import affine_term, factored_image, logical_axes_of
from .relation import Relation


def at_most(extent: str, limits: tuple, position: int) -> list[str]:
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
    values: dim.IslParamValues = {}
    for position in range(rank):
        begin, bound_begin = affine_term(offsets[position], f"o{position}")
        extent, bound_extent = affine_term(extents[position], f"e{position}")
        values.update(bound_begin)
        values.update(bound_extent)
        if bound_begin:
            guards.append(f"0 <= {begin}")
        if bound_extent:
            guards.append(f"1 <= {extent}")
            guards.extend(at_most(extent, limits, position))
        if (bound_begin or bound_extent) and position < len(within):
            whole, bound_whole = affine_term(within[position], f"w{position}")
            values.update(bound_whole)
            guards.append(f"{begin} + {extent} <= {whole}")
        if begin == "0":
            guards.append(f"0 <= d{position} < {extent}")
            continue
        guards.append(f"{begin} <= d{position} < {begin} + {extent}")
    prefix = declared(values)
    where = f" : {' and '.join(guards)}" if guards else ""
    reached = isl.map(f"{prefix}{{ [{domain}] -> [{domain}]{where} }}")
    whole = isl.map(f"{prefix}{{ [{domain}] -> [{domain}] }}")
    left = whole.subtract(reached).intersect_params(reached.params())
    return Relation(left, values), Relation(reached, values)


def normalised_rows(local: "types.Type", logical: "types.Type", first: int) -> tuple:
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
        elif expr.static_dim_value(extent) == 1:
            names.append("0")
        else:
            names.append(f"j{position}")
            guards.append(f"0 <= j{position} < {extent}")
    return tuple(extents), tuple(names), tuple(guards)


def control_leaves(type_: "types.Type") -> int:
    """How many numbers one operand carries for placing or sizing a window.

    A window is placed by one number per axis it is placed on, and an operand
    holding several of them carries several: a tuple of offsets is read once per
    leaf it holds, however its fields are nested.
    """
    return len(ir_utils.leaves_of(type_))


def control_space(rank: int, ctx, arg) -> "tuple[str, str, str]":
    """The domain, image and reach of one control operand's own coordinates.

    A tuple of numbers is indexed flat, one leaf however its fields are nested.
    A lone scalar's legal index set is the single point, at whatever positions
    its own view gives it.
    """
    domain = ", ".join(f"d{index}" for index in range(rank))
    stated = ctx.type_of(arg)
    if isinstance(stated, types.TupleType):
        return domain, "l", f"0 <= l < {control_leaves(stated)}"
    held = ctx.local_type_of(arg)
    return domain, ", ".join("0" for _ in range(len(getattr(held, "shape", ()) or ()))), ""


def control_read(rank: int, ctx, arg) -> Relation:
    """The control numbers one operand carries, each read once.

    The domain is the result's positions like every other boundary, because a
    reader applies one execution domain to all of them and a boundary with a rank
    of its own is one it cannot answer. What it reaches is one point per number,
    so a reader counts the numbers rather than believing an empty set.
    """
    domain, image, reach = control_space(rank, ctx, arg)
    where = f" : {reach}" if reach else ""
    return Relation(isl.map(f"{{ [{domain}] -> [{image}]{where} }}"))


def reached_at(
    rank: int,
    local: "types.Type",
    logical: "types.Type",
    reads: dict,
    free: tuple = (),
) -> Relation:
    """The coordinates one operand is reached at, stated per logical axis.

    An Op reasons in logical axes and a participant is indexed by the positions
    its layout made, so the expressions are given per axis and spread over the
    positions holding it. An axis named `free` is one whose coordinate is a value
    nobody has here because data determines it, so this may-access relation
    covers every coordinate that axis could legally name instead of guessing one.
    That keeps the answer bounded, and countable, without a carrier a reader has
    to special-case.
    """
    belongs = logical_axes_of(local, logical)
    stated = [reads.get(axis, "0") for axis in range(len(logical.shape))]
    for axis in free:
        stated[axis] = "0"
    image = factored_image(stated, local, logical)
    values: dim.IslParamValues = {}
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
    return Relation(
        isl.map(f"{declared(values)}{{ [{domain}] -> [{', '.join(image)}]{where} }}"),
        values,
        may_access=bool(free),
    )


def window_source(
    offsets: tuple,
    rank: int,
    local: "types.Type",
    logical: "types.Type",
    carried: dict,
    extents: tuple = (),
    limits: tuple = (),
) -> Relation:
    """One operand read at its own coordinates, from where a window put them.

    A window covers the operand's shape wherever it lands, so the coordinate read
    is the one reached shifted back to where the window starts. The shift is per
    logical axis, because that is what an offset is stated against, and only then
    spread over the positions this operand's own layout made -- which are not the
    result's. An axis whose extent is given holds the read to it, because an
    operand supplying more than the window takes is not read past it; an axis
    given ``None`` is covered whole and asks for no such guard.
    """
    values: dim.IslParamValues = {}
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
            guards.extend(at_most(extent, limits, axis))
        guards.append(
            f"0 <= {walked} - {begin} < {extent}" if begin != "0" else f"0 <= {walked} < {extent}"
        )
    domain = ", ".join(f"d{index}" for index in range(rank))
    image = ", ".join(factored_image(reads, local, logical))
    where = f" : {' and '.join(guards)}" if guards else ""
    return Relation(isl.map(f"{declared(values)}{{ [{domain}] -> [{image}]{where} }}"), values)


def declared(values: dim.IslParamValues) -> str:
    """The parameter list a relation built from *values* declares, or nothing."""
    return f"[{', '.join(values)}] -> " if values else ""


__all__ = [
    "control_leaves",
    "control_read",
    "normalised_rows",
    "placed_window",
    "reached_at",
    "window_source",
]
