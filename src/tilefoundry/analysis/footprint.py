"""Measure authored-loop buffer access without requiring a schedule."""

from __future__ import annotations

import math

import isl

from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry.access_relation import index_set

from .poly.affine import LoopAffineTerm


class _Unavailable(Exception):
    """An access that cannot be represented by this authored-loop model."""


def _static_loop_bound(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise _Unavailable


def _widest_allowed(
    access: isl.map, name: str, held: object
) -> LoopAffineTerm | None:
    """The value a parameter may take that reaches the most of its operand.

    A footprint is an upper bound, so a parameter nobody here can place takes
    whichever end of its legal range touches more: where a window sits does not
    change how much of it there is, but how long it is does. Both ends are the
    Op's own contract, read off the relation rather than guessed.
    """
    names = [
        access.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(access.dim(isl.dim_type.PARAM))
    ]
    space = f"[{', '.join(names)}] -> "
    probe = isl.set(f"{space}{{ [x] : x = {name} }}").intersect_params(access.params())
    ends = (probe.dim_min_val(0), probe.dim_max_val(0))
    if not all(end.is_int() for end in ends):
        return None
    box = index_set(tuple(held.shape)) if isinstance(held, TensorType) else None
    if box is None or box.dim(isl.dim_type.SET) != access.dim(isl.dim_type.OUT):
        least = ends[0].get_num_si()
        return LoopAffineTerm(None, 0, least, least)
    best: tuple[int, int] | None = None
    for value in sorted({end.get_num_si() for end in ends}):
        reach = access.intersect_params(
            isl.set(f"{space}{{ : {name} = {value} }}")
        ).range().intersect(box)
        amount = reach.coalesce().count_val()
        if not amount.is_int():
            return None
        if best is None or amount.get_num_si() > best[0]:
            best = (amount.get_num_si(), value)
    return None if best is None else LoopAffineTerm(None, 0, best[1], best[1])




def _packed_bytes(elements: int, bit_width: int) -> int:
    return math.ceil(elements * bit_width / 8)
