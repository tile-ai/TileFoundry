"""Unique addresses one wave touches in a cache-backed level.

The requested enclosing loops are held at their first iteration, the addresses
every unit of one wave reaches are unioned, and both loads and stores occupy the
level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import isl

from tilefoundry.ir.core import Call, Expr, get_metadata, value_labels
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import DType, TensorType, TupleType, Type
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import (
    ComposedLayout,
    Layout,
    Mesh,
    flatten,
    topology_axes,
    try_c_order_strides,
)
from tilefoundry.ir.visitor import expr_children
from tilefoundry.target.base import Target, UnsupportedCapabilityError
from tilefoundry.target.facts import ParallelCapacityFacts
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry.access_relation import leaves_of, projected
from tilefoundry.visitor_registry.contexts import CostContext

from .access import Access, AccessPrecision, resolve_access
from .facts import MemoryHierarchyFacts
from .iteration_scope import IterationScope, walk_scopes
from .loop_domain import induction_name
from .metadata import (
    Breakdown,
    Footprint,
    MemoryMetadata,
    ReuseWindow,
    Spread,
    TrafficBytes,
)


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


@dataclass(frozen=True)
class ReuseAxes:
    """Where a second read of one boundary's data can come from.

    ``time`` is the outermost loop dimension whose iterations reach the same
    addresses; ``space`` names the mesh coordinates the addresses do not vary
    with, and ``space_units`` counts the units mapping onto those addresses.
    Both axes absent means the data is read once.
    """

    time: int | None = None
    space: tuple[str, ...] = ()
    space_units: int = 1

    @property
    def window(self) -> int | None:
        """Return the loop dimension held for this reuse window.

        Space-only reuse uses Python's ``-1`` spelling for the innermost loop;
        the caller resolves it against the boundary's scope depth.
        """
        if self.time is not None:
            return self.time
        return -1 if self.space else None


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


def reuse_axes(
    access: Access,
    unit_access: Access | None,
    scope: IterationScope,
    mesh_params: tuple[tuple[str | None, Mesh, int], ...],
    *,
    wave_access: Access,
    wave_units: int,
    declared_units: int,
) -> ReuseAxes:
    """Name reuse axes and how many units share their reached addresses."""
    lineage: list[IterationScope] = []
    cursor: IterationScope | None = scope
    while cursor is not None:
        if isinstance(cursor.owner, LoopRegion):
            lineage.append(cursor)
        cursor = cursor.parent
    lineage.reverse()

    time = None
    for axis, loop_scope in enumerate(lineage):
        if loop_scope.trips() <= 1:
            continue
        held = _at_first_iteration(access.relation, axis + 1).range()
        released = _at_first_iteration(access.relation, axis).range()
        if held.is_equal(released):
            time = axis
            break

    if unit_access is None or not mesh_params:
        return ReuseAxes(time=time)

    stated_mesh_params = _mesh_parameters(scope)
    position = None
    if wave_units < declared_units and stated_mesh_params:
        position = _linear_position(stated_mesh_params)
        if position is None:
            return ReuseAxes(time=time)

    wave_relation = _at_first_iteration(wave_access.relation, scope.depth)
    unit_relation = _at_first_iteration(unit_access.relation, scope.depth)
    if position is not None:
        wave_relation = _restrict_to_wave(wave_relation, position, wave_units)
        unit_relation = _restrict_to_wave(unit_relation, position, wave_units)
    wave_reached = wave_relation.range()
    for parameter_name, _coordinate in stated_mesh_params:
        parameter = wave_reached.find_dim_by_name(isl.dim_type.PARAM, parameter_name)
        if parameter >= 0:
            wave_reached = wave_reached.project_out(isl.dim_type.PARAM, parameter, 1)

    space: list[str] = []
    shared_axes: list[int] = []
    for name, mesh, axis in mesh_params:
        shape = flatten(mesh.layout.shape)
        if not 0 <= axis < len(shape):
            continue
        extent = static_dim_value(shape[axis])
        if extent is None or extent <= 1:
            continue

        one_unit = unit_relation
        parameter = (
            -1
            if name is None
            else one_unit.find_dim_by_name(isl.dim_type.PARAM, name)
        )
        if parameter >= 0:
            parameter_values = one_unit.params().move_dims(
                isl.dim_type.SET, 0, isl.dim_type.PARAM, parameter, 1
            )
            if parameter_values.dim(isl.dim_type.PARAM):
                parameter_values = parameter_values.project_out(
                    isl.dim_type.PARAM,
                    0,
                    parameter_values.dim(isl.dim_type.PARAM),
                )
            first = parameter_values.dim_min_val(0)
            if not first.is_int():
                continue
            one_unit = one_unit.fix_val(isl.dim_type.PARAM, parameter, first)
        unit_reached = one_unit.range()
        for parameter_name, _coordinate in stated_mesh_params:
            coordinate = unit_reached.find_dim_by_name(
                isl.dim_type.PARAM, parameter_name
            )
            if coordinate >= 0:
                unit_reached = unit_reached.project_out(
                    isl.dim_type.PARAM, coordinate, 1
                )
        if not wave_reached.is_equal(unit_reached):
            continue

        axis_name = (
            mesh.names[axis]
            if axis < len(mesh.names)
            else ("x", "y", "z")[axis]
            if axis < 3
            else str(axis)
        )
        level_name = ""
        for level, axes in zip(mesh.topologies, topology_axes(mesh), strict=True):
            if axis in axes:
                level_name = getattr(level, "name", str(level))
                break
        space.append(f"{level_name}.{axis_name}" if level_name else axis_name)
        shared_axes.append(axis)

    if not shared_axes:
        return ReuseAxes(time=time)
    mesh = mesh_params[0][1]
    image = _layout_image(mesh)
    if image is None:
        return ReuseAxes(time=time)
    offset, shape, strides = image
    dimensions = ", ".join(f"d{axis}" for axis in range(len(shape)))
    points = isl.set(f"{{ [{dimensions}] }}")
    local = isl.local_space.from_space(points.get_space())
    for axis, extent in enumerate(shape):
        lower = isl.constraint.alloc_inequality(local).set_coefficient_si(
            isl.dim_type.SET, axis, 1
        )
        upper = (
            isl.constraint.alloc_inequality(local)
            .set_coefficient_si(isl.dim_type.SET, axis, -1)
            .set_constant_si(extent - 1)
        )
        points = points.add_constraint(lower).add_constraint(upper)
    if wave_units < declared_units:
        lower = isl.constraint.alloc_inequality(local).set_constant_si(offset)
        upper = isl.constraint.alloc_inequality(local).set_constant_si(
            wave_units - 1 - offset
        )
        for axis, stride in enumerate(strides):
            lower = lower.set_coefficient_si(isl.dim_type.SET, axis, stride)
            upper = upper.set_coefficient_si(isl.dim_type.SET, axis, -stride)
        points = points.add_constraint(lower).add_constraint(upper)
    for axis in reversed(range(len(shape))):
        if axis not in shared_axes:
            points = points.project_out(isl.dim_type.SET, axis, 1)
    units = cardinality(points)
    if units is None or units <= 1:
        return ReuseAxes(time=time)
    return ReuseAxes(time=time, space=tuple(space), space_units=units)


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
    for side in (scope.accesses.get("narrow", {}), scope.outputs.get("narrow", {})):
        recorded = side.get(id(call))
        if recorded is not None and recorded[0] is call:
            found.extend(recorded[1])
    return tuple(found)


def _missing_boundaries(
    call: Call,
    accesses: tuple[Access, ...],
    operands: tuple[TrafficBytes, ...],
    ctx: CostContext,
) -> tuple[ReachedAddresses, ...]:
    """Represent boundary-index gaps as uncounted lower-bound evidence."""
    recorded_inputs = {access.input_index for access in accesses if access.input_index is not None}
    recorded_outputs = {
        access.output_index for access in accesses if access.output_index is not None
    }
    try:
        output = ctx.local_type_of(call)
    except ValueError:
        return (ReachedAddresses(call, None, None, None, False),)
    output_count = len(output.fields) if isinstance(output, TupleType) else 1
    missing_inputs = (
        ReachedAddresses(call.args[index], None, None, None, False)
        for index in sorted(set(range(len(call.args))) - recorded_inputs)
        if operands[index].read > 0 or operands[index].write > 0
    )
    missing_outputs = (
        ReachedAddresses(call, index, None, None, False)
        for index in sorted(set(range(output_count)) - recorded_outputs)
        if operands[-1].read > 0 or operands[-1].write > 0
    )
    return (*missing_inputs, *missing_outputs)


def _uncounted_boundaries(call: Call, ctx: CostContext) -> tuple[ReachedAddresses, ...]:
    """Represent every boundary when no positional movement answer exists."""
    try:
        output = ctx.local_type_of(call)
    except ValueError:
        return (ReachedAddresses(call, None, None, None, False),)
    output_count = len(output.fields) if isinstance(output, TupleType) else 1
    return (
        *(ReachedAddresses(arg, None, None, None, False) for arg in call.args),
        *(ReachedAddresses(call, index, None, None, False) for index in range(output_count)),
    )


def reached_by(
    scope: IterationScope,
    call: Call,
    *,
    memory_level: str,
    wave_units: int,
    declared_units: int,
    operands: tuple[TrafficBytes, ...],
    ctx: CostContext,
    window: int,
) -> tuple[ReachedAddresses, ...] | None:
    """Return one Call's cache-backed addresses, or ``None`` for no stated wave."""
    mesh_parameters = _mesh_parameters(scope)
    position = None
    if wave_units < declared_units and mesh_parameters:
        position = _linear_position(mesh_parameters)
        if position is None:
            return None

    if len(operands) != len(call.args) + 1:
        return _uncounted_boundaries(call, ctx)

    refused = call in scope.refused.get("narrow", ())
    accesses = _call_accesses(scope, call)
    result: list[ReachedAddresses] = []
    for access in accesses:
        moved = operands[access.input_index] if access.input_index is not None else operands[-1]
        if moved.read <= 0 and moved.write <= 0:
            continue
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

        relation = _at_first_iteration(access.relation, window + 1)
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
    result.extend(_missing_boundaries(call, accesses, operands, ctx))
    return tuple(result)


def reuse_windows(
    root: IterationScope,
    *,
    memory_level: str,
    wave_units: int,
    declared_units: int,
    ctx: CostContext,
) -> tuple[ReuseWindow, ...]:
    """Describe one cache-residency window per buffer that is read again."""
    unit_ctx = ctx
    whole_ctx = replace(ctx, topology_level=None, topologies=())
    scopes = tuple(walk_scopes(root))
    meshes_by_call: dict[int, Mesh] = {}
    if isinstance(root.owner, Function) and root.owner.body is not None:
        pending: list[tuple[Expr, Mesh | None]] = [(root.owner.body, None)]
        visited: set[tuple[int, int | None]] = set()
        while pending:
            expr, current_mesh = pending.pop()
            visit_key = (id(expr), None if current_mesh is None else id(current_mesh))
            if visit_key in visited:
                continue
            visited.add(visit_key)
            if isinstance(expr, MeshRegion):
                pending.extend((arg, current_mesh) for arg in reversed(expr.args))
                pending.append((expr.body, expr.mesh))
                continue
            if isinstance(expr, Call) and current_mesh is not None:
                meshes_by_call.setdefault(id(expr), current_mesh)
            pending.extend((child, current_mesh) for child in reversed(expr_children(expr)))
    buffers: dict[int, Expr] = {}
    axes_by_buffer: dict[int, ReuseAxes] = {}
    time_scopes: dict[int, IterationScope | None] = {}
    space_units: dict[int, int] = {}

    for scope in scopes:
        stated_mesh_params = _mesh_parameters(scope)
        lineage: list[IterationScope] = []
        cursor: IterationScope | None = scope
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                lineage.append(cursor)
            cursor = cursor.parent
        lineage.reverse()

        for call, _recorded in scope.accesses.get("narrow", {}).values():
            moved = get_metadata(call, MemoryMetadata)
            if moved is None or len(moved.operands) != len(call.args) + 1:
                continue
            device_recorded = scope.accesses.get("device", {}).get(id(call))
            device_by_boundary = (
                {
                    (item.input_index, item.output_index): item
                    for item in device_recorded[1]
                }
                if device_recorded is not None and device_recorded[0] is call
                else {}
            )
            try:
                local_relations = projected(
                    scope.stated_relations(call, unit_ctx), call, unit_ctx
                )
            except (NotImplementedError, TypeError, ValueError, isl.Error):
                local_relations = None
            current_mesh = meshes_by_call.get(id(call))
            if current_mesh is None and stated_mesh_params:
                current_mesh = stated_mesh_params[0][1].target.mesh
            if current_mesh is None:
                mesh_params: tuple[tuple[str | None, Mesh, int], ...] = ()
            else:
                names_by_axis = {
                    axis: name
                    for name, coordinate in stated_mesh_params
                    if coordinate.target.mesh == current_mesh
                    and coordinate.args
                    and (axis := static_dim_value(coordinate.args[0])) is not None
                }
                mesh_params = tuple(
                    (names_by_axis.get(axis), current_mesh, axis)
                    for axis in range(len(flatten(current_mesh.layout.shape)))
                )
            for access in _call_accesses(scope, call):
                movement = (
                    moved.operands[access.input_index]
                    if access.input_index is not None
                    else moved.operands[-1]
                )
                if movement.read <= 0:
                    continue
                held = _boundary_type(access, call, whole_ctx)
                if held is None or memory_level not in {
                    str(leaf.storage) for leaf in leaves_of(held)
                }:
                    continue
                unit_access = access
                if local_relations is not None:
                    try:
                        if access.input_index is not None:
                            boundary = local_relations.inputs[access.input_index]
                            operand = call.args[access.input_index]
                            local_rank = boundary.pattern.relation.dim(isl.dim_type.OUT)
                            logical_rank = (
                                len(operand.type.shape)
                                if isinstance(operand.type, TensorType)
                                else local_rank
                            )
                            if local_rank == logical_rank:
                                projected_access = resolve_access(
                                    operand,
                                    boundary,
                                    scope,
                                    unit_ctx,
                                    input_index=access.input_index,
                                    narrow=False,
                                )
                                if projected_access is not None:
                                    unit_access = projected_access
                        elif access.output_index is not None:
                            boundary = local_relations.outputs[access.output_index]
                            local_rank = boundary.pattern.relation.dim(isl.dim_type.OUT)
                            output_type = _boundary_type(access, call, whole_ctx)
                            logical_rank = (
                                len(output_type.shape)
                                if isinstance(output_type, TensorType)
                                else local_rank
                            )
                            if local_rank == logical_rank:
                                projected_access = resolve_access(
                                    call,
                                    boundary,
                                    scope,
                                    unit_ctx,
                                    input_index=None,
                                    output_index=access.output_index,
                                    narrow=False,
                                )
                                if projected_access is not None:
                                    unit_access = projected_access
                    except (IndexError, NotImplementedError, TypeError, ValueError, isl.Error):
                        unit_access = access
                boundary_key = (access.input_index, access.output_index)
                axes = reuse_axes(
                    access,
                    unit_access,
                    scope,
                    mesh_params,
                    wave_access=device_by_boundary.get(boundary_key, access),
                    wave_units=wave_units,
                    declared_units=declared_units,
                )
                if axes.window is None:
                    continue

                key = id(access.buffer)
                buffers.setdefault(key, access.buffer)
                current = axes_by_buffer.get(key, ReuseAxes())
                candidate_time = lineage[axes.time] if axes.time is not None else None
                selected_time = time_scopes.get(key)
                if candidate_time is not None and (
                    selected_time is None or candidate_time.depth < selected_time.depth
                ):
                    selected_time = candidate_time
                merged_space = tuple(dict.fromkeys((*current.space, *axes.space)))
                axes_by_buffer[key] = ReuseAxes(
                    time=None if selected_time is None else selected_time.depth - 1,
                    space=merged_space,
                )
                time_scopes[key] = selected_time
                space_units[key] = max(space_units.get(key, 1), axes.space_units)

    if not buffers or ctx.scope is None:
        return ()
    cache = cached_level(ctx.scope.module.resolve_target().get_facts(MemoryHierarchyFacts))
    if cache is None or cache[1] != memory_level:
        return ()
    _cache_level, _backing_level, capacity = cache

    window_owners: dict[int | None, LoopRegion | None] = {}
    for key in buffers:
        time_scope = time_scopes.get(key)
        owner = time_scope.owner if time_scope is not None else None
        window_owners[id(owner) if owner is not None else None] = owner

    reaches_by_window: dict[int | None, tuple[ReachedAddresses, ...] | None] = {}
    for window_key, owner in window_owners.items():
        reached_items: list[ReachedAddresses] = []
        available = True
        for scope in scopes:
            loops = scope.enclosing_loops()
            if owner is None:
                window = scope.depth - 1
            else:
                positions = [index for index, loop in enumerate(loops) if loop is owner]
                if not positions:
                    continue
                window = positions[0]

            for call, _recorded in scope.accesses.get("narrow", {}).values():
                moved = get_metadata(call, MemoryMetadata)
                operands = () if moved is None else moved.operands
                reached = reached_by(
                    scope,
                    call,
                    memory_level=memory_level,
                    wave_units=wave_units,
                    declared_units=declared_units,
                    operands=operands,
                    ctx=whole_ctx,
                    window=window,
                )
                if reached is None:
                    available = False
                else:
                    reached_items.extend(reached)
            for call in scope.refused.get("narrow", ()):
                reached = reached_by(
                    scope,
                    call,
                    memory_level=memory_level,
                    wave_units=wave_units,
                    declared_units=declared_units,
                    operands=(),
                    ctx=whole_ctx,
                    window=window,
                )
                if reached is None:
                    available = False
                else:
                    reached_items.extend(reached)
        reaches_by_window[window_key] = tuple(reached_items) if available else None

    all_reached = merged(
        item
        for reached in reaches_by_window.values()
        if reached is not None
        for item in reached
    )
    distinct: dict[int, Expr] = {}
    for item in all_reached:
        if item.reached is not None:
            distinct.setdefault(id(item.buffer), item.buffer)
    for key, buffer in buffers.items():
        distinct.setdefault(key, buffer)
    labels = dict(zip(distinct, value_labels(distinct.values()), strict=True))

    footprints: dict[int | None, Footprint] = {}
    for key, reached in reaches_by_window.items():
        if reached is not None:
            footprints[key] = footprint_of(
                merged(reached), memory_level=memory_level, labels=labels
            )

    rows: list[ReuseWindow] = []
    for key in buffers:
        time_scope = time_scopes.get(key)
        owner = time_scope.owner if time_scope is not None else None
        counted = footprints.get(id(owner) if owner is not None else None)
        if counted is None:
            continue
        amounts = {
            name: sum(spread.total for _level, spread in breakdown.kinds)
            for name, breakdown in counted.buffers
        }
        label = labels[key]
        holds = sum(amounts.values())
        repeats = (1 if time_scope is None else time_scope.trips()) * space_units.get(key, 1)
        saves = max(0, repeats - 1) * amounts.get(label, 0)
        if saves == 0:
            continue
        rows.append(
            ReuseWindow(
                buffer=label,
                time=(
                    induction_name(time_scope.owner)
                    if time_scope is not None and isinstance(time_scope.owner, LoopRegion)
                    else ""
                ),
                space=",".join(axes_by_buffer[key].space),
                holds_bytes=holds,
                saves_bytes=saves,
                fits=holds < capacity,
                complete=counted.complete,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.saves_bytes, reverse=True))


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


def footprint_of(
    items: Iterable[ReachedAddresses],
    *,
    memory_level: str,
    labels: Mapping[int, str],
) -> Footprint:
    """Count unioned addresses and pack their element widths into bytes."""
    items = tuple(items)
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


def wave_of(
    module: Module,
    target: Target,
    topology_level: str | None,
) -> tuple[int, int] | None:
    """Return ``(wave_units, declared_units)``, or ``None`` when unstated."""
    try:
        capacity = target.get_facts(ParallelCapacityFacts, topology_level)
        declared_units = static_dim_value(module.resolve_topology(capacity.topology).size)
    except (UnsupportedCapabilityError, ValueError):
        return None
    if declared_units is None:
        return None
    return min(declared_units, capacity.parallel_units), declared_units


def cached_level(facts: MemoryHierarchyFacts) -> tuple[str, str, int] | None:
    """Return the first stated cache and the addressable level it backs."""
    stated = sorted(
        (
            (level.name, level.capacity_bytes)
            for level in facts.implicit_levels
            if level.capacity_bytes is not None
        ),
        key=lambda item: item[0],
    )
    if not stated:
        return None
    cache_level, capacity = stated[0]
    return cache_level, facts.backing_level(cache_level), capacity


__all__ = [
    "ReachedAddresses",
    "ReuseAxes",
    "cached_level",
    "footprint_of",
    "merged",
    "reached_by",
    "reuse_axes",
    "reuse_windows",
    "wave_of",
]
