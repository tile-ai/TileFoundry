"""Addresses a boundary reaches, and the unique bytes one wave touches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property

import isl

from tilefoundry import utils
from tilefoundry.ir import core, dim, expr
from tilefoundry.ir import utils as ir_utils

from .access import widest_allowed
from .precision import Precision
from .quantity import Breakdown, breakdown


@dataclass(frozen=True)
class ReachedAddresses:
    """One boundary's addresses before anything is counted."""

    target: core.SSAValue
    output_index: int | None
    dtype: expr.DType | None
    reached: isl.set | None
    precision: Precision
    must: isl.set | None = None
    may_coordinates: int | None = 0
    values: Mapping[str, object] = field(default_factory=dict)
    ownership: tuple[str, ...] = ()

    @cached_property
    def counted(self):
        return count_reached(self)

    def __post_init__(self) -> None:
        if (self.dtype is None) != (self.reached is None):
            raise ValueError("dtype and reached addresses must be present together")
        if self.must is None and self.reached is not None:
            object.__setattr__(self, "must", self.reached)


@dataclass(frozen=True)
class Footprint:
    """Unique bytes one wave touches, by source buffer and memory level."""

    groups: tuple[tuple[int, tuple[ReachedAddresses, ...]], ...] = ()

    @cached_property
    def merged_groups(self):
        return tuple((windows, merged(items)) for windows, items in self.groups)

    @cached_property
    def reached(self) -> tuple[ReachedAddresses, ...]:
        return merged(item for _, items in self.merged_groups for item in items)

    @cached_property
    def buffers(self) -> tuple[tuple[core.SSAValue, tuple[str, ...], Breakdown[int]], ...]:
        return counted_buffers(self)

    @cached_property
    def precision(self) -> Precision:
        return counted_precision(item for _, items in self.merged_groups for item in items)

    def bytes_at(self, level: str) -> tuple[int, int] | None:
        """One window's and every executed window's bytes, when all sit at *level*.

        Reads and writes are one count: they share one bandwidth.
        """
        kinds = tuple(item for *_, held in self.buffers for item in held.kinds)
        if any(name != level for name, _ in kinds):
            return None
        return sum(s.logical for _, s in kinds), sum(s.total for _, s in kinds)


def project_mesh_parameters(
    reached: isl.set, parameters: tuple[tuple[str, expr.Call], ...]
) -> isl.set:
    """Remove mesh-coordinate parameters after their units have been unioned."""
    for name, _ in parameters:
        axis = reached.find_dim_by_name(isl.dim_type.PARAM, name)
        if axis >= 0:
            reached = reached.project_out(isl.dim_type.PARAM, axis, 1)
    return reached


def merged(items: Iterable[ReachedAddresses]) -> tuple[ReachedAddresses, ...]:
    """Union addresses by source allocation and output field."""
    grouped: dict[tuple[core.SSAValue, tuple[str, ...], int | None], ReachedAddresses] = {}
    for item in items:
        key = (item.target, item.ownership, item.output_index)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = item
            continue
        precision = previous.precision.join(item.precision)
        if previous.reached is None:
            grouped[key] = replace(item, precision=precision)
        elif item.reached is None:
            grouped[key] = replace(previous, precision=precision)
        else:
            grouped[key] = replace(
                previous,
                reached=previous.reached.union(item.reached),
                must=previous.must.union(item.must),
                values={**previous.values, **item.values},
                may_coordinates=(
                    None
                    if previous.may_coordinates is None or item.may_coordinates is None
                    else previous.may_coordinates + item.may_coordinates
                ),
                precision=precision,
            )
    return tuple(grouped.values())


def footprint_of(items, *, windows=1) -> Footprint:
    """Retain one copy of the reached evidence; numeric views are derived."""
    return Footprint(((windows, tuple(items)),))


def count_reached(item):
    if item.reached is None:
        return 0, item.precision
    image = item.reached
    mesh_params = tuple(
        name
        for name, value in item.values.items()
        if isinstance(value, expr.Call) and isinstance(value.target, dim.MeshCoord)
    )
    bounded = item.precision
    for axis in reversed(range(image.dim(isl.dim_type.PARAM))):
        name = image.get_dim_name(isl.dim_type.PARAM, axis)
        if name not in mesh_params:
            parameter = image.params().move_dims(isl.dim_type.SET, 0, isl.dim_type.PARAM, axis, 1)
            determined = isl.map.from_range(parameter).is_single_valued()
            relevant = image.gist_params(image.params()).involves_dims(isl.dim_type.PARAM, axis, 1)
            if not determined and relevant:
                independent = image.project_out(isl.dim_type.PARAM, axis, 1).intersect_params(
                    image.params()
                )
                relevant = not independent.is_equal(image)
            if not determined and relevant:
                held = ir_utils.leaves_of(item.target.expr.type)[item.target.index]
                value = widest_allowed(isl.map.from_range(image), name, held)
                if value is not None:
                    image = image.fix_val(isl.dim_type.PARAM, axis, isl.val(value))
                bounded = bounded.join(Precision.UPPER_BOUND)
            image = image.project_out(isl.dim_type.PARAM, axis, 1)
    image = project_mesh_parameters(image, tuple((name, None) for name in mesh_params))
    amount = utils.cardinality(image)
    if amount is None:
        return 0, (
            Precision.UNKNOWN if item.precision is Precision.UNKNOWN else Precision.LOWER_BOUND
        )
    if item.may_coordinates is not None and not item.must.is_equal(item.reached):
        must = utils.cardinality(item.must, union=mesh_params)
        if must is not None:
            amount = min(amount, must + item.may_coordinates)
    return -(-(amount * item.dtype.bit_width) // 8), bounded


def counted_precision(items):
    precision = Precision.EXACT
    for item in items:
        precision = precision.join(item.counted[1])
    return precision


def counted_buffers(footprint):
    """Union by SSA identity and instance ownership before packing bytes."""
    groups, items = footprint.merged_groups, footprint.reached
    totals = {}
    for windows, group in groups:
        for item in group:
            key = item.target, item.ownership
            totals[key] = totals.get(key, 0) + item.counted[0] * windows
    logical = {}
    for item in items:
        if item.reached is not None:
            key = item.target, item.ownership
            logical[key] = logical.get(key, 0) + item.counted[0]
    entries = []
    for key, amount in logical.items():
        tgt, ownership = key
        level = str(ir_utils.leaves_of(tgt.expr.type)[tgt.index].storage)
        held = breakdown({level: totals[key]}, (), 0, logical={level: amount})
        entries.append((tgt, ownership, held))
    return tuple(entries)


__all__ = [
    "Footprint",
    "ReachedAddresses",
    "counted_buffers",
    "counted_precision",
    "footprint_of",
    "merged",
    "project_mesh_parameters",
]
