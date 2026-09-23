"""Unique addresses one wave touches in a cache-backed level.

The requested enclosing loops are held at their first iteration, the addresses
every unit of one wave reaches are unioned, and both loads and stores occupy the
level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.core import Call, Expr, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.loop_region import LoopRegion
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


def _loop_scopes(scope: IterationScope) -> tuple[IterationScope, ...]:
    """Return enclosing loop scopes in outer-to-inner order."""
    lineage: list[IterationScope] = []
    cursor: IterationScope | None = scope
    while cursor is not None:
        if isinstance(cursor.owner, LoopRegion):
            lineage.append(cursor)
        cursor = cursor.parent
    lineage.reverse()
    return tuple(lineage)


def _project_mesh_parameters(reached: isl.set, parameters: tuple[tuple[str, Call], ...]) -> isl.set:
    """Remove mesh-coordinate parameters after their units have been unioned."""
    for name, _coordinate in parameters:
        axis = reached.find_dim_by_name(isl.dim_type.PARAM, name)
        if axis >= 0:
            reached = reached.project_out(isl.dim_type.PARAM, axis, 1)
    return reached


@dataclass
class MovingBoundary:
    """One moving boundary at this level, with reached addresses cached by window."""

    scope: IterationScope
    call: Call
    access: Access
    space_wave_access: Access
    unit_access: Access
    dtype: DType
    label: str
    mesh: Mesh | None
    reads: bool
    wave_units: int
    wave_stated: bool
    mesh_parameters: tuple[tuple[str, Call], ...]
    axis_parameters: tuple[str | None, ...]
    position: tuple[int, tuple[tuple[str, int], ...]] | None
    _relations: dict[int, isl.map] = field(default_factory=dict, init=False, repr=False)
    _held: dict[int, isl.set] = field(default_factory=dict, init=False, repr=False)
    _reached: dict[int, isl.set] = field(default_factory=dict, init=False, repr=False)
    _space_wave_reached: dict[int, isl.set] = field(
        default_factory=dict, init=False, repr=False
    )
    _unit_relations: dict[int, isl.map] = field(default_factory=dict, init=False, repr=False)
    _unit_reached: dict[tuple[int, str | None], isl.set | None] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def exact(self) -> bool:
        """Whether this boundary's access relation is exact."""
        return self.access.precision is AccessPrecision.EXACT

    def _relation(self, window: int) -> isl.map:
        relation = self._relations.get(window)
        if relation is None:
            relation = _at_first_iteration(self.access.relation, window + 1)
            self._relations[window] = relation
        return relation

    def held(self, window: int) -> isl.set:
        """Return raw reached addresses, retaining every mesh parameter."""
        held = self._held.get(window)
        if held is None:
            held = self._relation(window).range()
            self._held[window] = held
        return held

    def reached(self, window: int) -> isl.set:
        """Return wave-unioned addresses at *window*, computing them once."""
        reached = self._reached.get(window)
        if reached is None:
            relation = self._relation(window)
            if self.position is not None:
                relation = _restrict_to_wave(relation, self.position, self.wave_units)
            reached = _project_mesh_parameters(relation.range(), self.mesh_parameters)
            self._reached[window] = reached
        return reached

    def _unit_relation(self, window: int) -> isl.map:
        relation = self._unit_relations.get(window)
        if relation is None:
            relation = _at_first_iteration(self.unit_access.relation, window + 1)
            self._unit_relations[window] = relation
        return relation

    def space_wave_reached(self, window: int) -> isl.set:
        """Return the device-view wave addresses used by the space test."""
        reached = self._space_wave_reached.get(window)
        if reached is None:
            relation = _at_first_iteration(
                self.space_wave_access.relation, window + 1
            )
            if self.position is not None:
                relation = _restrict_to_wave(relation, self.position, self.wave_units)
            reached = _project_mesh_parameters(relation.range(), self.mesh_parameters)
            self._space_wave_reached[window] = reached
        return reached

    def unit_reached(self, window: int, parameter_name: str | None) -> isl.set | None:
        """Return one feasible unit's addresses from the cached window relation."""
        key = (window, parameter_name)
        if key in self._unit_reached:
            return self._unit_reached[key]
        relation = self._unit_relation(window)
        if self.position is not None:
            relation = _restrict_to_wave(relation, self.position, self.wave_units)
        parameter = (
            -1
            if parameter_name is None
            else relation.find_dim_by_name(isl.dim_type.PARAM, parameter_name)
        )
        if parameter >= 0:
            values = relation.params().move_dims(
                isl.dim_type.SET, 0, isl.dim_type.PARAM, parameter, 1
            )
            if values.dim(isl.dim_type.PARAM):
                values = values.project_out(
                    isl.dim_type.PARAM, 0, values.dim(isl.dim_type.PARAM)
                )
            first = values.dim_min_val(0)
            if not first.is_int():
                self._unit_reached[key] = None
                return None
            relation = relation.fix_val(isl.dim_type.PARAM, parameter, first)
        reached = _project_mesh_parameters(relation.range(), self.mesh_parameters)
        self._unit_reached[key] = reached
        return reached


def time_axis(boundary: MovingBoundary) -> int | None:
    """Return the outermost loop whose iterations reach the same addresses."""
    for axis, loop_scope in enumerate(_loop_scopes(boundary.scope)):
        if loop_scope.trips() <= 1:
            continue
        if boundary.held(axis).is_equal(boundary.held(axis - 1)):
            return axis
    return None


def space_axes(
    boundary: MovingBoundary, wave: tuple[int, int]
) -> tuple[int, ...]:
    """Return mesh axes where one unit reaches the wave-unioned addresses."""
    if boundary.mesh is None or wave[0] <= 1:
        return ()
    window = boundary.scope.depth - 1
    wave_reached = boundary.space_wave_reached(window)
    shared = []
    for axis, parameter_name in enumerate(boundary.axis_parameters):
        shape = flatten(boundary.mesh.layout.shape)
        extent = static_dim_value(shape[axis]) if axis < len(shape) else None
        unit_reached = boundary.unit_reached(window, parameter_name)
        if extent is not None and extent > 1 and unit_reached is not None:
            if wave_reached.is_equal(unit_reached):
                shared.append(axis)
    return tuple(shared)


def axis_label(mesh: Mesh, axis: int) -> str:
    """Name a mesh axis, falling back to its factual numeric position."""
    level_name = ""
    for level, axes in zip(mesh.topologies, topology_axes(mesh), strict=True):
        if axis in axes:
            level_name = getattr(level, "name", str(level))
            break
    if axis < len(mesh.names):
        name = mesh.names[axis]
        return f"{level_name}.{name}" if level_name else name
    return f"{level_name}[{axis}]" if level_name else f"[{axis}]"


def shared_units(mesh: Mesh, axes: tuple[int, ...], wave: tuple[int, int]) -> int:
    """Count wave positions after projecting out axes that do not share data."""
    image = _layout_image(mesh)
    if image is None:
        return 1
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
    wave_units, declared_units = wave
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
        if axis not in axes:
            points = points.project_out(isl.dim_type.SET, axis, 1)
    units = cardinality(points)
    return units if units is not None and units > 1 else 1


def reuse_axes(boundary: MovingBoundary, wave: tuple[int, int]) -> ReuseAxes:
    """Name reuse axes and how many units share this boundary's addresses."""
    time = time_axis(boundary)
    axes = space_axes(boundary, wave)
    units = shared_units(boundary.mesh, axes, wave) if boundary.mesh and axes else 1
    if units <= 1:
        axes = ()
    return ReuseAxes(
        time=time,
        space=tuple(axis_label(boundary.mesh, axis) for axis in axes)
        if boundary.mesh
        else (),
        space_units=units,
    )


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


def _axis_parameters(
    scope: IterationScope, mesh: Mesh | None
) -> tuple[str | None, ...]:
    """Map each axis of the innermost mesh to its retained isl parameter."""
    if mesh is None:
        return ()
    names_by_axis = {
        axis: name
        for name, coordinate in _mesh_parameters(scope)
        if coordinate.target.mesh == mesh
        and coordinate.args
        and (axis := static_dim_value(coordinate.args[0])) is not None
    }
    return tuple(
        names_by_axis.get(axis) for axis in range(len(flatten(mesh.layout.shape)))
    )


def moving_boundaries(
    root: IterationScope,
    *,
    memory_level: str,
    wave: tuple[int, int],
    ctx: CostContext,
    labels: Mapping[int, str],
) -> tuple[MovingBoundary, ...]:
    """Collect boundaries that move bytes at *memory_level* in one scope walk."""
    whole = replace(ctx, topology_level=None, topologies=())
    wave_units, declared_units = wave
    found: list[MovingBoundary] = []
    for scope in walk_scopes(root):
        mesh_parameters = _mesh_parameters(scope)
        position = None
        if wave_units < declared_units and mesh_parameters:
            position = _linear_position(mesh_parameters)
        mesh = scope.enclosing_mesh()
        axis_parameters = _axis_parameters(scope, mesh)
        wave_stated = not (
            wave_units < declared_units and mesh_parameters and position is None
        )
        if not wave_stated:
            axis_parameters = ()
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
                unit_relations = projected(
                    scope.stated_relations(call, ctx), call, ctx
                )
            except (NotImplementedError, TypeError, ValueError, isl.Error):
                unit_relations = None
            for access in _call_accesses(scope, call):
                movement = (
                    moved.operands[access.input_index]
                    if access.input_index is not None
                    else moved.operands[-1]
                )
                if movement.read <= 0 and movement.write <= 0:
                    continue
                held = _boundary_type(access, call, whole)
                if held is None:
                    continue
                leaves = leaves_of(held)
                if memory_level not in {str(leaf.storage) for leaf in leaves}:
                    continue
                if len(leaves) != 1 or not isinstance(leaves[0], TensorType):
                    continue
                label = labels.get(id(access.buffer))
                if label is None:
                    continue
                boundary_key = (access.input_index, access.output_index)
                space_wave_access = device_by_boundary.get(boundary_key, access)
                unit_access = access
                if unit_relations is not None:
                    try:
                        if access.input_index is not None:
                            boundary = unit_relations.inputs[access.input_index]
                            operand = call.args[access.input_index]
                            local_rank = boundary.pattern.relation.dim(isl.dim_type.OUT)
                            logical_rank = (
                                len(operand.type.shape)
                                if isinstance(operand.type, TensorType)
                                else local_rank
                            )
                            if local_rank == logical_rank:
                                unit_access = (
                                    resolve_access(
                                        operand,
                                        boundary,
                                        scope,
                                        ctx,
                                        input_index=access.input_index,
                                        narrow=False,
                                    )
                                    or access
                                )
                        elif access.output_index is not None:
                            boundary = unit_relations.outputs[access.output_index]
                            local_rank = boundary.pattern.relation.dim(isl.dim_type.OUT)
                            logical_rank = (
                                len(held.shape)
                                if isinstance(held, TensorType)
                                else local_rank
                            )
                            if local_rank == logical_rank:
                                unit_access = (
                                    resolve_access(
                                        call,
                                        boundary,
                                        scope,
                                        ctx,
                                        input_index=None,
                                        output_index=access.output_index,
                                        narrow=False,
                                    )
                                    or access
                                )
                    except (IndexError, NotImplementedError, TypeError, ValueError, isl.Error):
                        unit_access = access
                found.append(
                    MovingBoundary(
                        scope=scope,
                        call=call,
                        access=access,
                        space_wave_access=space_wave_access,
                        unit_access=unit_access,
                        dtype=leaves[0].dtype,
                        label=label,
                        mesh=mesh,
                        reads=movement.read > 0,
                        wave_units=wave_units,
                        wave_stated=wave_stated,
                        mesh_parameters=mesh_parameters,
                        axis_parameters=axis_parameters,
                        position=position,
                    )
                )
    return tuple(found)


def _uncounted_movements(
    root: IterationScope,
    *,
    memory_level: str,
    ctx: CostContext,
) -> tuple[tuple[IterationScope, ReachedAddresses], ...]:
    """Keep completeness evidence that has no countable moving boundary."""
    whole = replace(ctx, topology_level=None, topologies=())
    found: list[tuple[IterationScope, ReachedAddresses]] = []
    for scope in walk_scopes(root):
        for call in scope.refused.get("narrow", ()):
            found.extend((scope, item) for item in _uncounted_boundaries(call, whole))
        for call, _recorded in scope.accesses.get("narrow", {}).values():
            moved = get_metadata(call, MemoryMetadata)
            operands = () if moved is None else moved.operands
            accesses = _call_accesses(scope, call)
            if len(operands) != len(call.args) + 1:
                found.extend((scope, item) for item in _uncounted_boundaries(call, whole))
                continue
            for access in accesses:
                movement = (
                    operands[access.input_index]
                    if access.input_index is not None
                    else operands[-1]
                )
                if movement.read <= 0 and movement.write <= 0:
                    continue
                held = _boundary_type(access, call, whole)
                if held is None:
                    continue
                leaves = leaves_of(held)
                if memory_level in {str(leaf.storage) for leaf in leaves} and (
                    len(leaves) != 1 or not isinstance(leaves[0], TensorType)
                ):
                    found.append(
                        (
                            scope,
                            ReachedAddresses(
                                access.buffer,
                                access.output_index,
                                None,
                                None,
                                False,
                            ),
                        )
                    )
            found.extend(
                (scope, item)
                for item in _missing_boundaries(call, accesses, operands, whole)
            )
    return tuple(found)


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


@dataclass
class _BufferReuse:
    """The selected reuse axes for one source buffer."""

    boundary: MovingBoundary
    time_scope: IterationScope | None
    space: tuple[str, ...]
    space_units: int


def _by_buffer(
    boundaries: tuple[MovingBoundary, ...], wave: tuple[int, int]
) -> dict[int, _BufferReuse]:
    """Select the outermost time axis and union space axes per buffer."""
    buffers: dict[int, _BufferReuse] = {}
    for boundary in boundaries:
        if not boundary.reads:
            continue
        axes = reuse_axes(boundary, wave)
        if axes.window is None:
            continue
        loops = _loop_scopes(boundary.scope)
        candidate = loops[axes.time] if axes.time is not None else None
        key = id(boundary.access.buffer)
        current = buffers.get(key)
        selected = None if current is None else current.time_scope
        if candidate is not None and (
            selected is None or candidate.depth < selected.depth
        ):
            selected = candidate
        space = tuple(
            dict.fromkeys((*(current.space if current else ()), *axes.space))
        )
        buffers[key] = _BufferReuse(
            boundary=boundary if current is None else current.boundary,
            time_scope=selected,
            space=space,
            space_units=max(axes.space_units, current.space_units if current else 1),
        )
    return buffers


def _window_index(
    scope: IterationScope, time_scope: IterationScope | None
) -> int | None:
    """Translate one selected loop scope into this boundary's loop position."""
    if time_scope is None:
        return scope.depth - 1
    for index, loop in enumerate(scope.enclosing_loops()):
        if loop is time_scope.owner:
            return index
    return None


def _window_footprints(
    boundaries: tuple[MovingBoundary, ...],
    uncounted: tuple[tuple[IterationScope, ReachedAddresses], ...],
    buffers: Mapping[int, _BufferReuse],
    *,
    memory_level: str,
) -> dict[int | None, Footprint | None]:
    """Count every moving buffer once in each distinct selected window."""
    labels = {
        id(boundary.access.buffer): boundary.label for boundary in boundaries
    }
    windows = {
        id(item.time_scope.owner) if item.time_scope is not None else None: item.time_scope
        for item in buffers.values()
    }
    footprints: dict[int | None, Footprint | None] = {}
    for key, time_scope in windows.items():
        reached: list[ReachedAddresses] = []
        for boundary in boundaries:
            window = _window_index(boundary.scope, time_scope)
            if window is None:
                continue
            if not boundary.wave_stated:
                footprints[key] = None
                break
            reached.append(
                ReachedAddresses(
                    boundary.access.buffer,
                    boundary.access.output_index,
                    boundary.dtype,
                    boundary.reached(window),
                    boundary.exact,
                )
            )
        if key in footprints:
            continue
        for scope, item in uncounted:
            if _window_index(scope, time_scope) is not None:
                reached.append(item)
        footprints[key] = footprint_of(
            merged(reached), memory_level=memory_level, labels=labels
        )
    return footprints


def _reuse_rows(
    buffers: Mapping[int, _BufferReuse],
    footprints: Mapping[int | None, Footprint | None],
    capacity: int,
) -> tuple[ReuseWindow, ...]:
    """Build sorted report rows from selected axes and counted windows."""
    rows: list[ReuseWindow] = []
    for item in buffers.values():
        key = id(item.time_scope.owner) if item.time_scope is not None else None
        counted = footprints.get(key)
        if counted is None:
            continue
        amounts = {
            name: sum(spread.total for _level, spread in breakdown.kinds)
            for name, breakdown in counted.buffers
        }
        holds = sum(amounts.values())
        repeats = (
            1 if item.time_scope is None else item.time_scope.trips()
        ) * item.space_units
        reuse = max(0, repeats - 1) * amounts.get(item.boundary.label, 0)
        if reuse == 0:
            continue
        rows.append(
            ReuseWindow(
                buffer=item.boundary.label,
                time=(
                    induction_name(item.time_scope.owner)
                    if item.time_scope is not None
                    and isinstance(item.time_scope.owner, LoopRegion)
                    else ""
                ),
                space=",".join(item.space),
                holds_bytes=holds,
                reuse_bytes=reuse,
                fits=holds < capacity,
                complete=counted.complete,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.reuse_bytes, reverse=True))


def reuse_windows(
    root: IterationScope,
    *,
    memory_level: str,
    wave_units: int,
    declared_units: int,
    ctx: CostContext,
    labels: Mapping[int, str],
) -> tuple[ReuseWindow, ...]:
    """Describe one cache-residency window per buffer that is read again."""
    wave = (wave_units, declared_units)
    boundaries = moving_boundaries(
        root, memory_level=memory_level, wave=wave, ctx=ctx, labels=labels
    )
    buffers = _by_buffer(boundaries, wave)
    if not buffers or ctx.scope is None:
        return ()
    cache = cached_level(ctx.scope.module.resolve_target().get_facts(MemoryHierarchyFacts))
    if cache is None or cache[1] != memory_level:
        return ()
    uncounted = _uncounted_movements(root, memory_level=memory_level, ctx=ctx)
    footprints = _window_footprints(
        boundaries, uncounted, buffers, memory_level=memory_level
    )
    return _reuse_rows(buffers, footprints, cache[2])


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
    "MovingBoundary",
    "ReachedAddresses",
    "ReuseAxes",
    "axis_label",
    "cached_level",
    "footprint_of",
    "merged",
    "moving_boundaries",
    "reached_by",
    "reuse_axes",
    "reuse_windows",
    "shared_units",
    "space_axes",
    "time_axis",
    "wave_of",
]
