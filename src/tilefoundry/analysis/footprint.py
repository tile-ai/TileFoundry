"""Unique addresses one wave touches in a cache-backed level.

The requested enclosing loops are held at their first iteration, the addresses
every unit of one wave reaches are unioned, and both loads and stores occupy the
level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import isl

from tilefoundry.ir.core import Call, get_metadata, value_label
from tilefoundry.ir.core.metadata import unique_labels
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.isl_interop import dim_to_isl_pw_aff
from tilefoundry.ir.types import DType, Mesh, TensorType, TupleType, Type
from tilefoundry.ir.types.int_tuple import repeat_like
from tilefoundry.ir.types.layout import ComposedLayout, flatten
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.target.base import Target, UnsupportedCapabilityError
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry.access_relation import AccessRelation, leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrameVisitor
from tilefoundry.visitor_registry.contexts import CostContext

from .access import (
    Access,
    AliasKind,
    BufferAlias,
    SSAValue,
    name_loop_parameters,
    restore_loop_dimensions,
    widest_allowed,
)
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
from .precision import AnalysisPrecision


@dataclass(frozen=True)
class ReachedAddresses:
    """One boundary's addresses before anything is counted."""

    target: SSAValue
    output_index: int | None
    dtype: DType | None
    reached: isl.set | None
    precision: AnalysisPrecision
    must: isl.set | None = None
    may_coordinates: int | None = 0
    values: Mapping[str, object] = field(default_factory=dict)
    ownership: tuple[str, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if (self.dtype is None) != (self.reached is None):
            raise ValueError("dtype and reached addresses must be present together")
        if self.must is None and self.reached is not None:
            object.__setattr__(self, "must", self.reached)


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


def _static_axes(mesh: Mesh) -> "tuple[int, tuple[int, ...], tuple[int, ...]] | None":
    """A mesh's device-numbered positions, when every one of them is a number."""
    try:
        axes = flatten(mesh.layout)
        offset = mesh.layout.offset if isinstance(mesh.layout, ComposedLayout) else 0
    except ValueError:
        return None
    shape = tuple(static_dim_value(extent) for extent in flatten(axes.shape))
    strides = tuple(static_dim_value(stride) for stride in flatten(axes.strides))
    if any(value is None for value in (*shape, *strides, static_dim_value(offset))):
        return None
    return offset, shape, strides


def _mesh_parameters(
    scope: IterationScope, relation: isl.map | None = None
) -> tuple[tuple[str, Call], ...]:
    """Mesh coordinates involved in this boundary, excluding unrelated retained facts."""
    return tuple(
        (name, value)
        for name, value in scope.domain_params.items()
        if isinstance(value, Call)
        and isinstance(value.target, MeshCoord)
        and (relation is None or _involves_parameter(relation, name))
    )


def _involves_parameter(relation: isl.map, name: str) -> bool:
    """Whether a parameter controls a coordinate rather than only its legal range."""
    relation = _union_loops(relation)
    position = relation.find_dim_by_name(isl.dim_type.PARAM, name)
    return position >= 0 and relation.gist_params(relation.params()).involves_dims(
        isl.dim_type.PARAM, position, 1
    )


def _linear_position(
    parameters: tuple[tuple[str, Call], ...],
) -> tuple[int, tuple[tuple[str, int], ...]] | None:
    """The layout-image position named by mesh-coordinate parameters."""
    if not parameters:
        return 0, ()
    mesh = parameters[0][1].target.mesh
    image = _static_axes(mesh)
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


def _loop_parameters(relation):
    return tuple(
        (axis, name)
        for axis in range(relation.dim(isl.dim_type.PARAM))
        if (name := relation.get_dim_name(isl.dim_type.PARAM, axis) or "").startswith("__tf_loop_")
    )


def _union_loops(relation):
    for axis, _name in reversed(_loop_parameters(relation)):
        relation = relation.project_out(isl.dim_type.PARAM, axis, 1)
    return relation


def _at_first_iteration(relation: isl.map, depth: int) -> isl.map:
    """Hold enclosing loop dimensions at their symbolic lexicographic first point."""
    rank = relation.dim(isl.dim_type.IN)
    own_rank = rank - depth
    loops = relation.domain().project_out(isl.dim_type.SET, depth, own_rank)
    first = loops.lexmin().insert_dims(isl.dim_type.SET, depth, own_rank)
    return relation.intersect_domain(first)


def _loop_scopes(scope: IterationScope) -> tuple[IterationScope, ...]:
    """Read loop facts from the common lexical contexts."""
    return tuple(
        reversed(
            [ctx.current for ctx in scope.context.ancestors() if isinstance(ctx.owner, LoopRegion)]
        )
    )


def _project_mesh_parameters(reached: isl.set, parameters: tuple[tuple[str, Call], ...]) -> isl.set:
    """Remove mesh-coordinate parameters after their units have been unioned."""
    for name, _coordinate in parameters:
        axis = reached.find_dim_by_name(isl.dim_type.PARAM, name)
        if axis >= 0:
            reached = reached.project_out(isl.dim_type.PARAM, axis, 1)
    return reached


def _reached_addresses(
    access: Access,
    relation: isl.map,
    dtype: DType,
    mesh_parameters: tuple[tuple[str, Call], ...],
    *,
    precision: AnalysisPrecision,
    values: Mapping[str, object] | None = None,
    ownership: tuple[str, ...] = (),
    label: str = "",
) -> ReachedAddresses:
    """Keep the possible image and the boundary's counting evidence together."""
    reached = relation.range()
    must = reached
    may_coordinates = 0
    if access.may_access:
        must = isl.set.empty(reached.get_space())
        may_coordinates = cardinality(relation.domain())
        if may_coordinates is None:
            may_coordinates = cardinality(reached)
    return ReachedAddresses(
        target=access.target,
        label=label,
        ownership=ownership,
        output_index=access.output_index,
        dtype=dtype,
        reached=reached,
        precision=precision,
        must=must,
        may_coordinates=may_coordinates,
        values={
            name: value
            for name, value in (values or {}).items()
            if reached.find_dim_by_name(isl.dim_type.PARAM, name) >= 0
        },
    )


@dataclass
class MovingBoundary:
    """One moving boundary at this level, with reached addresses cached by window."""

    scope: IterationScope
    call: Call
    access: Access
    space_wave_access: Access
    unit_access: Access
    dtype: DType
    base_label: str
    mesh: Mesh | None
    reads: bool
    wave_units: int
    wave_stated: bool
    mesh_parameters: tuple[tuple[str, Call], ...]
    axis_parameters: tuple[str | None, ...]
    position: tuple[int, tuple[tuple[str, int], ...]] | None
    ownership: tuple[str, ...] = ()

    _relations: dict[int, isl.map] = field(default_factory=dict, init=False, repr=False)
    _held: dict[int, isl.set] = field(default_factory=dict, init=False, repr=False)
    _reached: dict[int, isl.set] = field(default_factory=dict, init=False, repr=False)
    _space_wave_reached: dict[int, isl.set] = field(default_factory=dict, init=False, repr=False)
    _unit_relations: dict[int, isl.map] = field(default_factory=dict, init=False, repr=False)
    _unit_reached: dict[tuple[int, str | None], isl.set | None] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def depth(self):
        return self.scope.depth

    @property
    def loop_scopes(self):
        return _loop_scopes(self.scope)

    def _relation(self, window: int) -> isl.map:
        relation = self._relations.get(window)
        if relation is None:
            relation = _union_loops(_at_first_iteration(self.access.relation, window + 1))
            self._relations[window] = relation
        return relation

    def _wave_relation(self, window: int) -> isl.map:
        relation = self._relation(window)
        if self.position is not None:
            relation = _restrict_to_wave(relation, self.position, self.wave_units)
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
            reached = _project_mesh_parameters(
                self._wave_relation(window).range(), self.mesh_parameters
            )
            self._reached[window] = reached
        return reached

    def _unit_relation(self, window: int) -> isl.map:
        relation = self._unit_relations.get(window)
        if relation is None:
            relation = _union_loops(_at_first_iteration(self.unit_access.relation, window + 1))
            self._unit_relations[window] = relation
        return relation

    def space_wave_reached(self, window: int) -> isl.set:
        """Return the device-view wave addresses used by the space test."""
        reached = self._space_wave_reached.get(window)
        if reached is None:
            relation = _union_loops(
                _at_first_iteration(self.space_wave_access.relation, window + 1)
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
                values = values.project_out(isl.dim_type.PARAM, 0, values.dim(isl.dim_type.PARAM))
            first = values.dim_min_val(0)
            if not first.is_int():
                self._unit_reached[key] = None
                return None
            relation = relation.fix_val(isl.dim_type.PARAM, parameter, first)
        reached = _project_mesh_parameters(relation.range(), self.mesh_parameters)
        self._unit_reached[key] = reached
        return reached


def time_axis(boundary: MovingBoundary) -> int | None:
    """Return the outermost loop whose iterations reach the same addresses.

    First-iteration restrictions give held(axis) <= held(axis - 1), so only
    the reverse inclusion remains to prove equality.
    """
    value = (
        boundary.call.args[boundary.access.input_index]
        if boundary.access.input_index is not None
        else boundary.access.target.expr
    )
    target = boundary.access.target.expr
    if id(target) not in boundary.scope.variance and not getattr(target, "is_const", False):
        value = boundary.call
    for axis, loop_scope in enumerate(boundary.loop_scopes):
        if loop_scope.trips() <= 1 or (
            loop_scope.is_variant(boundary.access.target.expr) or loop_scope.is_variant(value)
        ):
            continue
        if boundary.held(axis - 1).is_subset(boundary.held(axis)):
            return axis
    return None


def space_axes(boundary: MovingBoundary, wave: tuple[int, int]) -> tuple[int, ...]:
    """Return mesh axes where one unit reaches the wave-unioned addresses."""
    if boundary.mesh is None or wave[0] <= 1:
        return ()
    window = boundary.depth - 1
    wave_reached = boundary.space_wave_reached(window)
    shared = []
    shape = flatten(flatten(boundary.mesh.layout).shape)
    for axis, parameter_name in enumerate(boundary.axis_parameters):
        extent = static_dim_value(shape[axis])
        unit_reached = boundary.unit_reached(window, parameter_name)
        if extent is not None and extent > 1 and unit_reached is not None:
            if wave_reached.is_equal(unit_reached):
                shared.append(axis)
    return tuple(shared)


def axis_label(mesh: Mesh, axis: int) -> str:
    """Name a mesh axis, falling back to its factual numeric position."""
    stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
    levels = flatten(
        tuple(
            repeat_like(mode, getattr(level, "name", str(level)))
            for mode, level in zip(stated.shape, mesh.topologies, strict=True)
        )
    )
    level_name = levels[axis] if axis < len(levels) else ""
    if axis < len(mesh.names):
        name = mesh.names[axis]
        return f"{level_name}.{name}" if level_name else name
    return f"{level_name}[{axis}]" if level_name else f"[{axis}]"


def shared_units(mesh: Mesh, axes: tuple[int, ...], wave: tuple[int, int]) -> int:
    """Count wave positions after projecting out axes that do not share data."""
    image = _static_axes(mesh)
    if image is None:
        return 1
    offset, shape, strides = image
    dimensions = ", ".join(f"d{axis}" for axis in range(len(shape)))
    points = isl.set(f"{{ [{dimensions}] }}")
    local = isl.local_space.from_space(points.get_space())
    for axis, extent in enumerate(shape):
        lower = isl.constraint.alloc_inequality(local).set_coefficient_si(isl.dim_type.SET, axis, 1)
        upper = (
            isl.constraint.alloc_inequality(local)
            .set_coefficient_si(isl.dim_type.SET, axis, -1)
            .set_constant_si(extent - 1)
        )
        points = points.add_constraint(lower).add_constraint(upper)
    wave_units, declared_units = wave
    if wave_units < declared_units:
        lower = isl.constraint.alloc_inequality(local).set_constant_si(offset)
        upper = isl.constraint.alloc_inequality(local).set_constant_si(wave_units - 1 - offset)
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
        space=tuple(axis_label(boundary.mesh, axis) for axis in axes) if boundary.mesh else (),
        space_units=units,
    )


def _boundary_type(access: Access, call: Call, ctx: CostContext) -> Type | None:
    """Recover the source Type for one input or indexed output boundary."""
    held = ctx.local_type_of(access.target.expr)
    fields = leaves_of(held)
    return fields[access.target.index] if 0 <= access.target.index < len(fields) else None


def _call_accesses(scope: IterationScope, call: Call, view: str = "narrow") -> tuple[Access, ...]:
    """The input and output boundaries recorded for one Call in a view."""
    found: list[Access] = []
    for side in (scope.accesses.get(view, {}), scope.outputs.get(view, {})):
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
        return (
            ReachedAddresses(SSAValue(call, 0), None, None, None, AnalysisPrecision.LOWER_BOUND),
        )
    output_count = len(output.fields) if isinstance(output, TupleType) else 1
    missing_inputs = (
        ReachedAddresses(
            SSAValue(call.args[index], 0), None, None, None, AnalysisPrecision.LOWER_BOUND
        )
        for index in sorted(set(range(len(call.args))) - recorded_inputs)
        if operands[index].read > 0 or operands[index].write > 0
    )
    missing_outputs = (
        ReachedAddresses(SSAValue(call, index), index, None, None, AnalysisPrecision.LOWER_BOUND)
        for index in sorted(set(range(output_count)) - recorded_outputs)
        if operands[-1].read > 0 or operands[-1].write > 0
    )
    return (*missing_inputs, *missing_outputs)


def _uncounted_boundaries(
    call: Call,
    ctx: CostContext,
    precision: AnalysisPrecision = AnalysisPrecision.LOWER_BOUND,
) -> tuple[ReachedAddresses, ...]:
    """Represent every boundary when no positional movement answer exists."""
    try:
        output = ctx.local_type_of(call)
    except ValueError:
        return (ReachedAddresses(SSAValue(call, 0), None, None, None, precision),)
    output_count = len(output.fields) if isinstance(output, TupleType) else 1
    return (
        *(ReachedAddresses(SSAValue(arg, 0), None, None, None, precision) for arg in call.args),
        *(
            ReachedAddresses(SSAValue(call, index), index, None, None, precision)
            for index in range(output_count)
        ),
    )


def _axis_parameters(scope: IterationScope, mesh: Mesh | None) -> tuple[str | None, ...]:
    """Map each wave-mesh axis to its retained isl parameter."""
    if mesh is None:
        return ()
    names_by_axis = {}
    for name, coordinate in _mesh_parameters(scope):
        axis = static_dim_value(coordinate.args[0])
        if axis is None:
            continue
        start = 0
        for level in separate(coordinate.target.mesh):
            rank = len(flatten(flatten(level.layout).shape))
            if level == mesh and start <= axis < start + rank:
                names_by_axis[axis - start] = name
            start += rank
    return tuple(
        names_by_axis.get(axis) for axis in range(len(flatten(flatten(mesh.layout).shape)))
    )


def _wave_mesh(scope: IterationScope, topology_level: str) -> Mesh | None:
    """Find the enclosing mesh, including the definition's checked outer mesh."""
    for ctx in scope.context.ancestors():
        mesh = (
            ctx.owner.mesh
            if isinstance(ctx.owner, MeshRegion)
            else (ctx.current_mesh if ctx.parent is None else None)
        )
        if mesh is not None:
            for component in separate(mesh):
                topology = component.topologies[0]
                if getattr(topology, "name", topology) == topology_level:
                    return component
    return None


def _one_time_fill(scope: IterationScope, operands: tuple[TrafficBytes, ...]) -> bool:
    """A write-only initializer outside loops streams once, without cache reuse."""
    return scope.depth == 0 and all(item.read == 0 for item in operands)


def _shared_origin(
    scope: IterationScope | None, topology_level: str, wave: tuple[int, int]
) -> bool:
    """A material definition made by multiple wave units is not shared data.

    Registered storage-sharing edges have already resolved the origin, so a
    view of an enclosing definition does not become fresh within this mesh.
    """
    mesh = None if scope is None else _wave_mesh(scope, topology_level)
    if mesh is None:
        return True
    axes = tuple(range(len(flatten(flatten(mesh.layout).shape))))
    return shared_units(mesh, axes, wave) <= 1


def _source_origin(scope, target, definitions):
    """Treat called material producers as private to their executing units."""
    if isinstance(target, Call) and id(target) not in scope.variance:
        return definitions.get(id(target), scope)
    return definitions.get(id(target))


def moving_boundaries(
    root: IterationScope,
    *,
    memory_level: str,
    wave: tuple[int, int],
    ctx: CostContext,
    labels: Mapping[SSAValue, str],
    resolved=(),
) -> tuple[MovingBoundary, ...]:
    """Collect boundaries that move bytes at *memory_level* in one scope walk.

    Definition-time source entries already name each boundary's material result.
    """
    whole = replace(ctx, topology_level=None, topologies=())
    wave_units, declared_units = wave
    topology_level = ctx.topology_level or ctx.frame.target.get_facts(TopologyFacts).parallel_level
    definitions = {value_id: scope for scope in walk_scopes(root) for value_id in scope.relations}
    definitions.update((id(call), scope) for scope, call, _reached in resolved)
    found: list[MovingBoundary] = []
    for scope in walk_scopes(root):
        mesh = _wave_mesh(scope, topology_level)
        axis_parameters = _axis_parameters(scope, mesh)
        for call, _recorded in scope.accesses.get("narrow", {}).values():
            moved = get_metadata(call, MemoryMetadata)
            if moved is None or len(moved.operands) != len(call.args) + 1:
                continue
            if _one_time_fill(scope, moved.operands):
                continue
            device_recorded = scope.accesses.get("device", {}).get(id(call))
            device_by_boundary = (
                {
                    (item.input_index, item.output_index, item.target.index): item
                    for item in device_recorded[1]
                }
                if device_recorded is not None and device_recorded[0] is call
                else {}
            )
            unit_by_boundary = {
                (item.input_index, item.output_index, item.target.index): item
                for item in _call_accesses(scope, call, "unit")
            }
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
                label = labels.get(access.target)
                if label is None:
                    continue
                boundary_key = (access.input_index, access.output_index, access.target.index)
                space_wave_access = device_by_boundary.get(boundary_key, access)
                unit_access = unit_by_boundary.get(boundary_key)
                if unit_access is None:
                    continue
                mesh_parameters = _mesh_parameters(scope, access.relation)
                position = (
                    _linear_position(mesh_parameters)
                    if wave_units < declared_units and mesh_parameters
                    else None
                )
                wave_stated = not (
                    wave_units < declared_units and mesh_parameters and position is None
                )
                found.append(
                    MovingBoundary(
                        scope=scope,
                        call=call,
                        access=access,
                        space_wave_access=space_wave_access,
                        unit_access=unit_access,
                        dtype=leaves[0].dtype,
                        base_label=label,
                        mesh=mesh,
                        reads=movement.read > 0,
                        wave_units=wave_units,
                        wave_stated=wave_stated,
                        mesh_parameters=mesh_parameters,
                        axis_parameters=axis_parameters
                        if wave_stated
                        and _shared_origin(
                            _source_origin(scope, access.target.expr, definitions),
                            topology_level,
                            wave,
                        )
                        else (),
                        position=position,
                    )
                )
    for scope, call, items in resolved:
        for item in items:
            if item.reached is None or item.dtype is None:
                continue
            relation = isl.map.from_range(item.reached).insert_dims(isl.dim_type.IN, 0, scope.depth)
            relation = relation.intersect_domain(scope.domain)
            relation = restore_loop_dimensions(
                relation, (loop.owner.induction_var for loop in _loop_scopes(scope))
            )
            access = Access(None, relation, item.target, precision=item.precision)
            mesh = _wave_mesh(scope, topology_level)
            parameters = tuple(
                (name, value)
                for name, value in {**scope.domain_params, **item.values}.items()
                if isinstance(value, Call)
                and isinstance(value.target, MeshCoord)
                and _involves_parameter(relation, name)
            )
            found.append(
                MovingBoundary(
                    scope=scope,
                    call=call,
                    access=access,
                    space_wave_access=access,
                    unit_access=access,
                    dtype=item.dtype,
                    base_label=item.label,
                    ownership=item.ownership,
                    mesh=mesh,
                    reads=item.output_index is None,
                    wave_units=wave_units,
                    wave_stated=True,
                    mesh_parameters=parameters,
                    axis_parameters=_axis_parameters(scope, mesh)
                    if _shared_origin(
                        _source_origin(scope, item.target.expr, definitions),
                        topology_level,
                        wave,
                    )
                    else (),
                    position=None,
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
        refused = scope.refused.get("narrow", frozenset()) | scope.refused.get("unit", frozenset())
        for call in refused:
            found.extend(
                (scope, item)
                for item in _uncounted_boundaries(call, whole, AnalysisPrecision.UNKNOWN)
            )
        for call, _recorded in scope.accesses.get("narrow", {}).values():
            if ctx.frame is not None and CallFrameVisitor.callee_frame(call, ctx.frame) is not None:
                continue
            if call in refused:
                continue
            moved = get_metadata(call, MemoryMetadata)
            operands = () if moved is None else moved.operands
            accesses = _call_accesses(scope, call)
            if len(operands) != len(call.args) + 1:
                found.extend((scope, item) for item in _uncounted_boundaries(call, whole))
                continue
            if _one_time_fill(scope, operands):
                continue
            for access in accesses:
                movement = (
                    operands[access.input_index] if access.input_index is not None else operands[-1]
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
                                access.target,
                                access.output_index,
                                None,
                                None,
                                AnalysisPrecision.LOWER_BOUND,
                            ),
                        )
                    )
            found.extend(
                (scope, item)
                for item in _missing_boundaries(
                    call, _call_accesses(scope, call, "unit"), operands, whole
                )
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
    capture_names: Mapping,
    window: int,
) -> tuple[ReachedAddresses, ...] | None:
    """Return one Call's cache-backed addresses, or ``None`` for no stated wave."""
    refused = call in scope.refused.get("narrow", ())
    if len(operands) != len(call.args) + 1:
        return _uncounted_boundaries(
            call,
            ctx,
            AnalysisPrecision.UNKNOWN if refused else AnalysisPrecision.LOWER_BOUND,
        )
    if _one_time_fill(scope, operands):
        return ()

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
                    target=access.target,
                    output_index=access.output_index,
                    dtype=None,
                    reached=None,
                    precision=AnalysisPrecision.LOWER_BOUND,
                )
            )
            continue

        mesh_parameters = _mesh_parameters(scope, access.relation)
        position = (
            _linear_position(mesh_parameters)
            if wave_units < declared_units and mesh_parameters
            else None
        )
        if wave_units < declared_units and mesh_parameters and position is None:
            return None
        relation, loop_values = name_loop_parameters(
            access.relation, (loop.owner.induction_var for loop in _loop_scopes(scope))
        )
        relation = relation.project_out(isl.dim_type.IN, 0, scope.depth)
        if position is not None:
            relation = _restrict_to_wave(relation, position, wave_units)
        result.append(
            _reached_addresses(
                access,
                relation,
                leaves[0].dtype,
                mesh_parameters,
                precision=AnalysisPrecision.UNKNOWN if refused else access.precision,
                values={**scope.domain_params, **loop_values},
            )
        )
    result.extend(_missing_boundaries(call, accesses, operands, ctx))
    return tuple(replace(item, label=source_label(item.target, capture_names)) for item in result)


@dataclass
class _BufferReuse:
    """The selected reuse axes for one source buffer."""

    boundary: MovingBoundary
    time_scope: IterationScope | None
    space: tuple[str, ...]
    space_units: int


def _by_buffer(
    boundaries: tuple[MovingBoundary, ...], wave: tuple[int, int]
) -> dict[tuple[SSAValue, tuple[str, ...]], _BufferReuse]:
    """Select the outermost time axis and union space axes per buffer."""
    buffers: dict[tuple[SSAValue, tuple[str, ...]], _BufferReuse] = {}
    for boundary in boundaries:
        if not boundary.reads:
            continue
        axes = reuse_axes(boundary, wave)
        if axes.window is None:
            continue
        loops = boundary.loop_scopes
        candidate = loops[axes.time] if axes.time is not None else None
        key = boundary.access.target, boundary.ownership
        current = buffers.get(key)
        choose = candidate is not None and (
            current is None
            or current.time_scope is None
            or candidate.depth < current.time_scope.depth
        )
        selected = candidate if choose else current.time_scope if current else None
        space = tuple(dict.fromkeys((*(current.space if current else ()), *axes.space)))
        buffers[key] = _BufferReuse(
            boundary=boundary if current is None else current.boundary,
            time_scope=selected,
            space=space,
            space_units=max(axes.space_units, current.space_units if current else 1),
        )
    return buffers


def _window_footprints(boundaries, uncounted, buffers, *, memory_level):
    """Count local moving boundaries in each selected local loop window."""
    footprints = {}
    for time_scope in {item.time_scope for item in buffers.values()}:
        reached = []
        for boundary in boundaries:
            window = (
                boundary.depth - 1
                if time_scope is None
                else next(
                    (axis for axis, loop in enumerate(boundary.loop_scopes) if loop is time_scope),
                    None,
                )
            )
            if window is None:
                continue
            if not boundary.wave_stated:
                footprints[time_scope] = None
                break
            reached.append(
                _reached_addresses(
                    boundary.access,
                    boundary._wave_relation(window),
                    boundary.dtype,
                    boundary.mesh_parameters,
                    precision=boundary.access.precision,
                    values={**boundary.scope.domain_params, **dict(boundary.mesh_parameters)},
                    ownership=boundary.ownership,
                    label=boundary.base_label,
                )
            )
        if time_scope in footprints:
            continue
        reached.extend(
            item
            for scope, item in uncounted
            if time_scope is None or time_scope in _loop_scopes(scope)
        )
        footprints[time_scope] = Footprint(((1, merged(reached)),))
    return footprints


def _reuse_rows(
    buffers: Mapping[tuple[SSAValue, tuple[str, ...]], _BufferReuse],
    footprints: Mapping[IterationScope | None, Footprint | None],
    capacity: int,
) -> tuple[ReuseWindow, ...]:
    """Build sorted report rows from selected axes and counted windows."""
    rows: list[ReuseWindow] = []
    for item in buffers.values():
        counted = footprints.get(item.time_scope)
        if counted is None:
            continue
        amounts = {
            name: sum(spread.logical for _level, spread in breakdown.kinds)
            for name, breakdown in counted.buffers
        }
        labels = footprint_labels(counted.reached)
        label = "/".join((*item.boundary.ownership, labels[item.boundary.access.target]))
        holds = sum(amounts.values())
        repeats = (1 if item.time_scope is None else item.time_scope.trips()) * item.space_units
        reuse = max(0, repeats - 1) * amounts.get(label, 0)
        if reuse == 0:
            continue
        rows.append(
            ReuseWindow(
                buffer=label,
                time=(
                    induction_name(item.time_scope.owner)
                    if item.time_scope is not None and isinstance(item.time_scope.owner, LoopRegion)
                    else ""
                ),
                space=",".join(item.space),
                holds_bytes=holds,
                reuse_bytes=reuse,
                fits=holds < capacity,
                precision=counted.precision,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.reuse_bytes, reverse=True))


def merged(items: Iterable[ReachedAddresses]) -> tuple[ReachedAddresses, ...]:
    """Union addresses by source allocation and output field."""
    grouped: dict[tuple[SSAValue, tuple[str, ...], int | None], ReachedAddresses] = {}
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


def _first_loops(image):
    names = tuple(name for _axis, name in _loop_parameters(image))
    first = image.params()
    for name in names:
        first = first.move_dims(
            isl.dim_type.SET,
            first.dim(isl.dim_type.SET),
            isl.dim_type.PARAM,
            first.find_dim_by_name(isl.dim_type.PARAM, name),
            1,
        )
    first = first.lexmin()
    for name in reversed(names):
        first = first.move_dims(
            isl.dim_type.PARAM,
            first.dim(isl.dim_type.PARAM),
            isl.dim_type.SET,
            first.find_dim_by_name(isl.dim_type.SET, name),
            1,
        )
    return _union_loops(image.intersect_params(first))


def _window_item(item):
    if item.reached is None:
        return item

    window = _union_loops if item.ownership else _first_loops
    image = window(item.reached)
    return replace(
        item, reached=image, must=image if item.must.is_equal(item.reached) else window(item.must)
    )


def _count(item):
    if item.reached is None:
        return 0, item.precision
    image = _first_loops(item.reached)
    image = _project_mesh_parameters(
        image,
        tuple(
            (name, value)
            for name, value in item.values.items()
            if isinstance(value, Call) and isinstance(value.target, MeshCoord)
        ),
    )
    bounded = item.precision
    for axis in reversed(range(image.dim(isl.dim_type.PARAM))):
        name = image.get_dim_name(isl.dim_type.PARAM, axis)
        if name is not None and name.startswith("__tf_binding_"):
            held = leaves_of(item.target.expr.type)[item.target.index]
            value = widest_allowed(isl.map.from_range(image), name, held)
            if value is not None:
                image = image.fix_val(isl.dim_type.PARAM, axis, isl.val(value))
            image = image.project_out(isl.dim_type.PARAM, axis, 1)
            bounded = bounded.join(AnalysisPrecision.UPPER_BOUND)
    amount = cardinality(image.coalesce())
    if amount is None:
        return 0, (
            AnalysisPrecision.UNKNOWN
            if item.precision is AnalysisPrecision.UNKNOWN
            else AnalysisPrecision.LOWER_BOUND
        )
    if item.may_coordinates is not None and not item.must.is_equal(item.reached):
        must = cardinality(item.must.coalesce())
        if must is not None:
            amount = min(amount, must + item.may_coordinates)
    return -(-(amount * item.dtype.bit_width) // 8), bounded


def counted_precision(items):
    precision = AnalysisPrecision.EXACT
    for item in items:
        precision = precision.join(_count(item)[1])
    return precision


def source_label(target: SSAValue, capture_names: Mapping) -> str:
    """Choose the declared name, first capture name, or existing value label."""
    root = target.expr
    named = root if getattr(root, "name", None) else capture_names.get(root, root)
    return value_label(named) or ""


def footprint_labels(items):
    """Disambiguate source bases together, then format flattened tensor leaves."""
    items = tuple(items)
    roots = {}
    for item in items:
        roots.setdefault(item.target.expr, item.label)
    labels = dict(zip(roots, unique_labels(list(roots.values())), strict=True))
    return {
        item.target: labels[item.target.expr]
        + (f"[{item.target.index}]" if isinstance(item.target.expr.type, TupleType) else "")
        for item in items
    }


def counted_buffers(footprint):
    """Union by SSA identity and instance ownership before packing bytes."""
    groups = tuple(
        (windows, merged(_window_item(item) for item in group))
        for windows, group in footprint.groups
    )
    items = merged(item for _windows, group in groups for item in group)
    labels = footprint_labels(items)
    counts = {}

    def count(item):
        if id(item) not in counts:
            counts[id(item)] = _count(item)[0]
        return counts[id(item)]

    totals = {}
    for windows, group in groups:
        for item in group:
            key = item.target, item.ownership
            totals[key] = totals.get(key, 0) + count(item) * windows
    logical, counted = {}, {}
    for item in items:
        if item.reached is not None:
            key = item.target, item.ownership
            logical[key] = logical.get(key, 0) + count(item)
            counted[key] = item
    entries = []
    for key, item in counted.items():
        held = leaves_of(item.target.expr.type)[item.target.index]
        entries.append(
            (
                "/".join((*item.ownership, labels[item.target])),
                Breakdown(
                    (
                        (
                            str(held.storage),
                            Spread(total=totals[key], logical_groups=(((), logical[key]),)),
                        ),
                    )
                ),
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def resolve_footprint(footprint, ctx):
    """Substitute reached coordinates through one leaf call boundary."""
    if footprint is None:
        return None
    from .metadata import BindingValuation, LoopNestValuation  # noqa: PLC0415

    binding = ctx.request_valuation(BindingValuation)
    loops = ctx.request_valuation(LoopNestValuation)
    domain = loops.domain()
    depth = domain.dim(isl.dim_type.SET)
    groups = []
    for windows, items in footprint.groups:
        resolved = []
        for item in items:
            if item.reached is None:
                resolved.append(item)
                continue

            def project_image(image):
                for axis in reversed(range(len(item.ownership))):
                    image = image.set_dim_name(isl.dim_type.SET, axis, f"__tf_instance_{axis}")
                    image = image.move_dims(
                        isl.dim_type.PARAM, image.dim(isl.dim_type.PARAM), isl.dim_type.SET, axis, 1
                    )
                source = BufferAlias(
                    item.target,
                    AccessRelation(isl.map.from_range(image), item.values),
                    AliasKind.FORWARD,
                )
                projected, precision = binding.project(source)
                relation = projected.relation.relation
                values = dict(projected.relation.values)
                relation = relation.project_out(isl.dim_type.IN, 0, depth)
                for axis in range(len(item.ownership)):
                    parameter = relation.find_dim_by_name(
                        isl.dim_type.PARAM, f"__tf_instance_{axis}"
                    )
                    relation = relation.move_dims(
                        isl.dim_type.OUT, axis, isl.dim_type.PARAM, parameter, 1
                    )
                image = relation.range()
                return projected.target, image, values, precision

            target, image, values, precision = project_image(item.reached)
            must = image if item.must.is_equal(item.reached) else project_image(item.must)[1]
            projected = replace(
                item,
                target=target,
                reached=image,
                must=must,
                precision=item.precision.join(precision),
                values=values,
            )
            if target != item.target:
                projected = replace(projected, label=source_label(target, ctx.root.capture_names))
            resolved.append(projected)
        groups.append((windows, tuple(resolved)))
    return Footprint(tuple(groups))


def qualify_instances(footprint, ctx):
    """Prefix weight address images with the actual slot index, never its label."""
    if footprint is None:
        return None
    target = ctx.call.target
    slot = target.args[0] if isinstance(target, Call) else target
    if not isinstance(slot, ModuleList):
        return footprint
    index = target.args[1] if isinstance(target, Call) else 0
    scope = ctx.current
    coords = {
        id(loop.owner.induction_var): scope.domain.get_dim_name(isl.dim_type.SET, axis)
        for axis, loop in enumerate(_loop_scopes(scope))
    }
    coords.update(
        {param: coords[id(root)] for param, root in ctx.loop_aliases.items() if id(root) in coords}
    )
    bounded = AnalysisPrecision.EXACT
    values = {}
    try:
        index_map = dim_to_isl_pw_aff(index, values, coords=coords or {-1: "dummy"}).as_map()
        if not coords:
            index_map = index_map.project_out(isl.dim_type.IN, 0, 1)
        index_map = index_map.intersect_domain(scope.domain)
        unknown = index_map.dim(isl.dim_type.PARAM) > 0
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        unknown = True
    if unknown:
        count = len(ctx.module.children[slot.key].elements)
        index_map = isl.map.from_domain(scope.domain).add_dims(isl.dim_type.OUT, 1)
        index_map = index_map.lower_bound_si(isl.dim_type.OUT, 0, 0).upper_bound_si(
            isl.dim_type.OUT, 0, count - 1
        )
        bounded = AnalysisPrecision.UPPER_BOUND
    groups = []
    for windows, items in footprint.groups:
        qualified = []
        for item in items:
            if item.reached is None or not (
                item.ownership
                or any(
                    item.target.expr is formal and formal.is_const
                    for formal in ctx.callee.function.params
                )
            ):
                qualified.append(item)
                continue
            relation = isl.map.from_range(item.reached).insert_dims(isl.dim_type.IN, 0, scope.depth)
            relation = relation.intersect_domain(scope.domain)
            relation = restore_loop_dimensions(
                relation, (loop.owner.induction_var for loop in _loop_scopes(scope))
            )
            relation = index_map.flat_range_product(relation)
            relation, _loop_values = name_loop_parameters(
                relation, (loop.owner.induction_var for loop in _loop_scopes(scope))
            )
            relation = relation.project_out(isl.dim_type.IN, 0, scope.depth)
            image = relation.range()
            qualified.append(
                replace(
                    item,
                    ownership=(slot.key, *item.ownership),
                    reached=image,
                    must=image,
                    precision=item.precision.join(bounded),
                )
            )
        groups.append((windows, tuple(qualified)))
    return Footprint(tuple(groups))


def reuse_windows(root, *, memory_level, wave, ctx, capacity, labels, resolved=()):
    boundaries = moving_boundaries(
        root, memory_level=memory_level, wave=wave, ctx=ctx, labels=labels, resolved=resolved
    )
    buffers = _by_buffer(boundaries, wave)
    footprints = _window_footprints(
        boundaries,
        _uncounted_movements(root, memory_level=memory_level, ctx=ctx),
        buffers,
        memory_level=memory_level,
    )
    return _reuse_rows(buffers, footprints, capacity)


def wave_of(
    module: Module,
    target: Target,
    topology_level: str | None,
    *,
    topologies: tuple[Topology, ...] | None = None,
) -> tuple[int, int] | None:
    """Return ``(wave_units, declared_units)``, or ``None`` when unstated."""
    try:
        facts = target.get_facts(TopologyFacts)
        level = facts.parallel() if topology_level is None else facts.level(topology_level)
        if level is None or level.max_physical_units is None:
            return None
        declared_units = static_dim_value(
            (
                module.resolve_topology(level.name)
                if topologies is None
                else next(item for item in topologies if item.name == level.name)
            ).size
        )
    except (UnsupportedCapabilityError, ValueError):
        return None
    if declared_units is None:
        return None
    return min(declared_units, level.max_physical_units), declared_units


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
    "shared_units",
    "space_axes",
    "time_axis",
    "wave_of",
]
