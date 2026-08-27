"""Measure authored-loop buffer access without requiring a schedule."""

from __future__ import annotations

import math

import isl

from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.shard_layout import split_target_axes
from tilefoundry.visitor_registry.access_relation import index_set

from .poly.affine import LoopAffineTerm


class _Unavailable(Exception):
    """An access that cannot be represented by this authored-loop model."""


def _local_type(type_: object) -> object:
    """Narrow every Split axis while preserving the tensor's logical rank."""
    if not isinstance(type_, TensorType):
        return type_
    layout = shard_layout_of(type_.layout)
    if layout is None:
        return type_
    local = list(type_.shape)
    for mesh_axis, tensor_axis in enumerate(split_target_axes(layout, type_.shape)):
        if tensor_axis is None:
            continue
        extent = layout.mesh.layout.shape[mesh_axis]
        if extent is None:
            local[tensor_axis] = 1
            continue
        size = local[tensor_axis]
        if not isinstance(size, int) or isinstance(size, bool) or size % extent != 0:
            raise _Unavailable
        local[tensor_axis] = size // extent
    return TensorType(shape=tuple(local), dtype=type_.dtype, layout=None, storage=type_.storage)


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
