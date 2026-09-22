"""Unique addresses one wave touches at one iteration in a cache-backed level.

Every enclosing loop is held at its first iteration, the addresses every unit
of one wave reaches are unioned, and both loads and stores occupy the level.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import isl

from tilefoundry.ir.core import Call, Expr, value_labels
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import DType, TensorType, TupleType, Type
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import (
    ComposedLayout,
    Layout,
    Mesh,
    flatten,
    try_c_order_strides,
)
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry.access_relation import leaves_of
from tilefoundry.visitor_registry.contexts import CostContext

from .access import Access, AccessPrecision
from .iteration_scope import IterationScope
from .metadata import Breakdown, Footprint, Spread


@dataclass(frozen=True)
class ReachedAddresses:
    """One boundary's addresses before anything is counted."""

    buffer: Expr
    output_index: int | None
    dtype: DType | None
    reached: isl.set | None
    exact: bool

    def __post_init__(self) -> None:
        if (self.dtype is None) != (self.reached is None):
            raise ValueError("dtype and reached addresses must be present together")


def _layout_image(mesh: Mesh) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    """Return a statically stated ``(offset, shape, strides)`` image."""
    layout = mesh.layout
    if isinstance(layout, Layout):
        stated = layout
        offset = 0
    elif (
        isinstance(layout, ComposedLayout)
        and layout.inner is None
        and isinstance(layout.outer, Layout)
    ):
        stated = layout.outer
        offset = layout.offset
    else:
        return None

    shape = flatten(stated.shape)
    if stated.strides is None:
        strides = try_c_order_strides(shape)
        if strides is None:
            return None
    else:
        strides = tuple(static_dim_value(value) for value in flatten(stated.strides))
        if any(value is None for value in strides):
            return None

    static_shape = tuple(static_dim_value(value) for value in shape)
    static_offset = static_dim_value(offset)
    if (
        static_offset is None
        or any(value is None for value in static_shape)
        or len(static_shape) != len(strides)
    ):
        return None
    return (
        static_offset,
        tuple(value for value in static_shape if value is not None),
        tuple(value for value in strides if value is not None),
    )


def _mesh_parameters(scope: IterationScope) -> tuple[tuple[str, Call], ...]:
    """Mesh-coordinate parameters retained by this scope's loop domain."""
    return tuple(
        (name, value)
        for name, value in scope.domain_params.items()
        if isinstance(value, Call) and isinstance(value.target, MeshCoord)
    )


def _linear_position(
    parameters: tuple[tuple[str, Call], ...],
) -> tuple[int, tuple[tuple[str, int], ...]] | None:
    """The layout-image position named by mesh-coordinate parameters."""
    if not parameters:
        return 0, ()
    mesh = parameters[0][1].target.mesh
    image = _layout_image(mesh)
    if image is None:
        return None
    offset, shape, strides = image
    terms: list[tuple[str, int]] = []
    for name, coordinate in parameters:
        if coordinate.target.mesh != mesh or not coordinate.args:
            return None
        axis = static_dim_value(coordinate.args[0])
        if axis is None or not 0 <= axis < len(shape):
            return None
        terms.append((name, strides[axis]))
    return offset, tuple(terms)


def _restrict_to_wave(
    relation: isl.map,
    position: tuple[int, tuple[tuple[str, int], ...]],
    wave_units: int,
) -> isl.map:
    """Constrain one layout-image position to ``[0, wave_units)``."""
    offset, terms = position
    context = relation.params()
    local = isl.local_space.from_space(context.get_space())
    lower = isl.constraint.alloc_inequality(local).set_constant_si(offset)
    upper = isl.constraint.alloc_inequality(local).set_constant_si(wave_units - 1 - offset)
    for name, stride in terms:
        axis = context.find_dim_by_name(isl.dim_type.PARAM, name)
        if axis < 0:
            continue
        lower = lower.set_coefficient_si(isl.dim_type.PARAM, axis, stride)
        upper = upper.set_coefficient_si(isl.dim_type.PARAM, axis, -stride)
    return relation.intersect_params(context.add_constraint(lower).add_constraint(upper))


def _at_first_iteration(relation: isl.map, depth: int) -> isl.map:
    """Hold enclosing loop dimensions at their symbolic lexicographic first point."""
    rank = relation.dim(isl.dim_type.IN)
    own_rank = rank - depth
    loops = relation.domain().project_out(isl.dim_type.SET, depth, own_rank)
    first = loops.lexmin().insert_dims(isl.dim_type.SET, depth, own_rank)
    return relation.intersect_domain(first)


def _boundary_type(access: Access, call: Call, ctx: CostContext) -> Type | None:
    """Recover the source Type for one input or indexed output boundary."""
    if access.output_index is None:
        return ctx.local_type_of(access.buffer)
    output = ctx.local_type_of(call)
    if isinstance(output, TupleType):
        index = access.output_index
        return output.fields[index] if 0 <= index < len(output.fields) else None
    return output if access.output_index == 0 else None


def _call_accesses(scope: IterationScope, call: Call) -> tuple[Access, ...]:
    """The narrow input and output boundaries recorded for one Call."""
    found: list[Access] = []
    for side in (scope.accesses["narrow"], scope.outputs["narrow"]):
        recorded = side.get(id(call))
        if recorded is not None and recorded[0] is call:
            found.extend(recorded[1])
    return tuple(found)


def _missing_boundaries(
    call: Call,
    accesses: tuple[Access, ...],
    ctx: CostContext,
) -> tuple[ReachedAddresses, ...]:
    """Represent boundary-index gaps as uncounted lower-bound evidence."""
    recorded_inputs = {access.input_index for access in accesses if access.input_index is not None}
    recorded_outputs = {
        access.output_index for access in accesses if access.output_index is not None
    }
    output = ctx.local_type_of(call)
    output_count = len(output.fields) if isinstance(output, TupleType) else 1
    missing_inputs = (
        ReachedAddresses(call.args[index], None, None, None, False)
        for index in sorted(set(range(len(call.args))) - recorded_inputs)
    )
    missing_outputs = (
        ReachedAddresses(call, index, None, None, False)
        for index in sorted(set(range(output_count)) - recorded_outputs)
    )
    return (*missing_inputs, *missing_outputs)


def reached_by(
    scope: IterationScope,
    call: Call,
    *,
    memory_level: str,
    wave_units: int,
    declared_units: int,
    ctx: CostContext,
) -> tuple[ReachedAddresses, ...] | None:
    """Return one Call's cache-backed addresses, or ``None`` for no stated wave."""
    mesh_parameters = _mesh_parameters(scope)
    position = None
    if wave_units < declared_units and mesh_parameters:
        position = _linear_position(mesh_parameters)
        if position is None:
            return None

    refused = call in scope.refused.get("narrow", ())
    accesses = _call_accesses(scope, call)
    result: list[ReachedAddresses] = []
    for access in accesses:
        held = _boundary_type(access, call, ctx)
        if held is None:
            continue
        leaves = leaves_of(held)
        if memory_level not in {str(leaf.storage) for leaf in leaves}:
            continue
        if len(leaves) != 1 or not isinstance(leaves[0], TensorType):
            result.append(
                ReachedAddresses(
                    buffer=access.buffer,
                    output_index=access.output_index,
                    dtype=None,
                    reached=None,
                    exact=False,
                )
            )
            continue

        relation = _at_first_iteration(access.relation, scope.depth)
        if position is not None:
            relation = _restrict_to_wave(relation, position, wave_units)
        reached = relation.range()
        for name, _coordinate in mesh_parameters:
            axis = reached.find_dim_by_name(isl.dim_type.PARAM, name)
            if axis >= 0:
                reached = reached.project_out(isl.dim_type.PARAM, axis, 1)
        result.append(
            ReachedAddresses(
                buffer=access.buffer,
                output_index=access.output_index,
                dtype=leaves[0].dtype,
                reached=reached,
                exact=access.precision is AccessPrecision.EXACT and not refused,
            )
        )
    if not refused:
        result.extend(_missing_boundaries(call, accesses, ctx))
    return tuple(result)


def merged(items: Iterable[ReachedAddresses]) -> tuple[ReachedAddresses, ...]:
    """Union addresses by source allocation and output field."""
    grouped: dict[tuple[int, int | None], ReachedAddresses] = {}
    for item in items:
        key = (id(item.buffer), item.output_index)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = item
            continue
        if previous.reached is None and item.reached is None:
            grouped[key] = ReachedAddresses(
                previous.buffer, previous.output_index, None, None, False
            )
            continue
        if previous.reached is None:
            grouped[key] = ReachedAddresses(
                item.buffer,
                item.output_index,
                item.dtype,
                item.reached,
                False,
            )
            continue
        if item.reached is None:
            grouped[key] = ReachedAddresses(
                previous.buffer,
                previous.output_index,
                previous.dtype,
                previous.reached,
                False,
            )
            continue
        if previous.dtype != item.dtype:
            grouped[key] = ReachedAddresses(
                previous.buffer, previous.output_index, None, None, False
            )
            continue
        grouped[key] = ReachedAddresses(
            buffer=previous.buffer,
            output_index=previous.output_index,
            dtype=previous.dtype,
            reached=previous.reached.union(item.reached),
            exact=previous.exact and item.exact,
        )
    return tuple(grouped.values())


def footprint_of(items: Iterable[ReachedAddresses], *, memory_level: str) -> Footprint:
    """Count unioned addresses and pack their element widths into bytes."""
    items = tuple(items)
    distinct: dict[int, Expr] = {}
    for item in items:
        distinct.setdefault(id(item.buffer), item.buffer)
    labels = dict(zip(distinct, value_labels(distinct.values()), strict=True))

    totals: dict[tuple[str, str], int] = {}
    complete = True
    for item in items:
        if item.reached is None:
            complete = False
            continue
        amount = cardinality(item.reached.coalesce())
        if amount is None or item.dtype is None:
            complete = False
            continue
        packed = -(-(amount * item.dtype.bit_width) // 8)
        key = (labels[id(item.buffer)], memory_level)
        totals[key] = totals.get(key, 0) + packed
        if not item.exact:
            complete = False

    buffers = tuple(
        (
            name,
            Breakdown(((level, Spread(size, size, ())),)),
        )
        for (name, level), size in sorted(totals.items())
    )
    return Footprint(buffers=buffers, complete=complete)


__all__ = ["ReachedAddresses", "footprint_of", "merged", "reached_by"]
