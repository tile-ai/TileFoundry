"""Unique addresses one wave touches in a cache-backed level.

The requested enclosing loops are held at their first iteration, the addresses
every unit of one wave reaches are unioned, and both loads and stores occupy the
level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property

import isl

from tilefoundry.ir.core import Call, Expr, get_metadata, value_labels
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import DType, Mesh, TensorType, TupleType, Type
from tilefoundry.ir.types.int_tuple import repeat_like
from tilefoundry.ir.types.layout import ComposedLayout, flatten
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.utils import leaves_of, static_dim_value
from tilefoundry.target.base import Target, UnsupportedCapabilityError
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry.access_relation import AccessRelation
from tilefoundry.visitor_registry.contexts import CostContext

from .access import Access, widest_allowed
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
from .visitor import AnalyzeContext


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

    @cached_property
    def counted(self):
        return _count(self)

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
    scope: IterationScope, relation: isl.map | None = None, *, values=None
) -> tuple[tuple[str, Call], ...]:
    """Mesh coordinates involved in this boundary, excluding unrelated retained facts."""
    return tuple(
        (name, value)
        for name, value in (scope.domain.values if values is None else values).items()
        if isinstance(value, Call)
        and isinstance(value.target, MeshCoord)
        and (relation is None or relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0)
    )


def _wave_position(relation, parameters):
    image = relation.range()
    mesh_names = {name for name, _value in parameters}
    for axis in reversed(range(image.dim(isl.dim_type.PARAM))):
        if image.get_dim_name(isl.dim_type.PARAM, axis) not in mesh_names:
            image = image.project_out(isl.dim_type.PARAM, axis, 1)
    shape = image.gist_params(image.params())
    involved = tuple(
        (name, value)
        for name, value in parameters
        if (axis := shape.find_dim_by_name(isl.dim_type.PARAM, name)) >= 0
        and shape.involves_dims(isl.dim_type.PARAM, axis, 1)
    )
    return _linear_position(involved)


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


def _loop_scopes(scope: IterationScope) -> tuple[IterationScope, ...]:
    """Read loop facts from the common lexical contexts."""
    return scope.loop_scopes


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
) -> ReachedAddresses:
    """Keep the possible image and the boundary's counting evidence together."""
    reached = relation.range()
    must = reached
    may_coordinates = 0
    if access.may_access:
        must = isl.set.empty(reached.get_space())
        may_coordinates = cardinality(
            relation.domain(), union=tuple(name for name, _value in mesh_parameters)
        )
        if may_coordinates is None:
            may_coordinates = cardinality(
                reached, union=tuple(name for name, _value in mesh_parameters)
            )
    return ReachedAddresses(
        target=access.target,
        ownership=ownership,
        output_index=access.output_index,
        dtype=dtype,
        reached=reached,
        precision=precision.join(AnalysisPrecision.UPPER_BOUND)
        if access.relation.may_access and not access.may_access
        else precision,
        must=must,
        may_coordinates=may_coordinates,
        values=access.relation._result(isl.map.from_range(reached), values or {}).values,
    )


@dataclass
class MovingBoundary:
    """One moving boundary at this level, with reached addresses cached by window."""

    scope: IterationScope
    dependency: Expr
    access: Access
    space_wave_access: Access
    unit_access: Access
    dtype: DType
    mesh: Mesh | None
    reads: bool
    wave_units: int
    wave_stated: bool
    mesh_parameters: tuple[tuple[str, Call], ...]
    axis_parameters: tuple[str | None, ...]
    position: tuple[int, tuple[tuple[str, int], ...]] | None
    ownership: tuple[str, ...] = ()
    loops: tuple[IterationScope, ...] = ()

    _views: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def depth(self):
        return len(self.loop_scopes)

    @property
    def loop_scopes(self):
        return self.scope.loop_scopes

    def view(self, kind, window, axis=None):
        key = kind, window, axis
        if key in self._views:
            return self._views[key]
        access = {
            "held": self.access,
            "wave": self.space_wave_access,
            "unit": self.unit_access,
            "relation": self.access,
        }[kind]
        loops = self.loop_scopes
        relation = access.relation.constrain(_first(loops[: window + 1]))
        relation = relation.project(tuple(s.owner.induction_var for s in self.loops or loops))
        if self.position is not None:
            relation = relation._result(
                _restrict_to_wave(relation.relation, self.position, self.wave_units)
            )
        if kind == "unit" and axis is not None:
            index = relation.relation.find_dim_by_name(isl.dim_type.PARAM, axis)
            if index >= 0:
                values = relation.relation.params().move_dims(
                    isl.dim_type.SET, 0, isl.dim_type.PARAM, index, 1
                )
                values = values.project_out(isl.dim_type.PARAM, 0, values.dim(isl.dim_type.PARAM))
                first = values.dim_min_val(0)
                if not first.is_int():
                    self._views[key] = None
                    return None
                relation = relation._result(
                    relation.relation.fix_val(isl.dim_type.PARAM, index, first)
                )
        result = (
            relation.relation
            if kind == "relation"
            else _project_mesh_parameters(relation.relation.range(), self.mesh_parameters)
        )
        self._views[key] = result
        return result


def time_axis(boundary: MovingBoundary) -> int | None:
    """Return the outermost loop whose iterations reach the same addresses.

    First-iteration restrictions give held(axis) <= held(axis - 1), so only
    the reverse inclusion remains to prove equality.
    """
    for axis, loop_scope in enumerate(boundary.loop_scopes):
        if loop_scope.trips() <= 1 or (
            loop_scope.is_variant(boundary.access.target.expr)
            or loop_scope.is_variant(boundary.dependency)
        ):
            continue
        if boundary.view("held", axis - 1).is_subset(boundary.view("held", axis)):
            return axis
    return None


def space_axes(boundary: MovingBoundary, wave: tuple[int, int]) -> tuple[int, ...]:
    """Return mesh axes where one unit reaches the wave-unioned addresses."""
    if boundary.mesh is None or wave[0] <= 1:
        return ()
    window = boundary.depth - 1
    wave_reached = boundary.view("wave", window)
    shared = []
    shape = flatten(flatten(boundary.mesh.layout).shape)
    for axis, parameter_name in enumerate(boundary.axis_parameters):
        extent = static_dim_value(shape[axis])
        unit_reached = boundary.view("unit", window, parameter_name)
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


def _call_accesses(
    scope: IterationScope, call: Call, view: str = "narrow", *, path=None
) -> tuple[Access, ...]:
    """The input and output boundaries recorded for one Call in a view."""
    found: list[Access] = []
    for side in (scope.access.accesses.get(view, {}), scope.access.outputs.get(view, {})):
        recorded = side.get(path or (SSAValue(call),))
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
    values = {
        name: value
        for records in (scope.access.accesses, scope.access.outputs)
        for _call, accesses in records.get("device", {}).values()
        for access in accesses
        for name, value in access.relation.values.items()
    }
    for name, coordinate in _mesh_parameters(scope, values=values):
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
    for ctx in scope.ancestors():
        mesh = (
            ctx.owner.mesh
            if isinstance(ctx.owner, MeshRegion)
            else (ctx.mesh if ctx.parent is None else None)
        )
        if mesh is not None:
            for component in separate(mesh):
                topology = component.topologies[0]
                if getattr(topology, "name", topology) == topology_level:
                    return component
    return None


def _access_scopes(scope, path, ctx):
    """Resolve lexical environments along an already recorded access path."""
    origin = scope.at(path)
    result = list(origin.loop_scopes)
    caller = origin.caller_scope
    while caller is not None:
        result[:0] = caller.loop_scopes
        caller = caller.caller_scope
    return origin, tuple(result)


def _first(loops):
    result = AccessRelation(isl.map("{ [] -> [] }"))
    for scope in loops:
        result = result.constrain(scope.first((scope,)))
    return result


def _one_time_fill(loops, operands: tuple[TrafficBytes, ...]) -> bool:
    """A write-only initializer outside loops streams once, without cache reuse."""
    return not loops and all(item.read == 0 for item in operands)


def _shared_origin(scope, target, definitions, topology_level, wave) -> bool:
    """A material definition made by multiple wave units is not shared data.

    Registered storage-sharing edges have already resolved the origin, so a
    view of an enclosing definition does not become fresh within this mesh.
    """
    origin = definitions.get(
        id(target),
        scope
        if isinstance(target, Call) and SSAValue(target) not in scope.provenance.snapshot
        else None,
    )
    mesh = None if origin is None else _wave_mesh(origin, topology_level)
    if mesh is None:
        return True
    axes = tuple(range(len(flatten(flatten(mesh.layout).shape))))
    return shared_units(mesh, axes, wave) <= 1


def moving_boundaries(
    root: IterationScope,
    *,
    memory_level: str,
    wave: tuple[int, int],
    ctx: AnalyzeContext,
    labels: Mapping[SSAValue, str],
) -> tuple[MovingBoundary, ...]:
    """Collect boundaries that move bytes at *memory_level* in one scope walk.

    Definition-time source entries already name each boundary's material result.
    """
    whole = replace(ctx, topology_level=None, topologies=())
    wave_units, declared_units = wave
    topology_level = ctx.topology_level or ctx.frame.target.get_facts(TopologyFacts).parallel_level
    definitions = {
        value_id: scope
        for definition in ctx.definitions.values()
        for scope in walk_scopes(definition)
        for value_id in (
            *scope.access.relations,
            *(
                id(phi)
                for phi, initial, yielded in scope.owner.carries()
                if scope.provenance.lookup(SSAValue(initial)).root
                != scope.provenance.lookup(SSAValue(yielded)).root
            ),
        )
    }
    found: list[MovingBoundary] = []
    for scope in walk_scopes(root):
        mesh = _wave_mesh(scope, topology_level)
        axis_parameters = _axis_parameters(scope, mesh)
        for path, (call, _recorded) in scope.access.accesses.get("narrow", {}).items():
            _origin, loops = _access_scopes(scope, path, ctx)
            moved = get_metadata(call, MemoryMetadata)
            if moved is None or len(moved.operands) != len(call.args) + 1:
                continue
            if _one_time_fill(loops, moved.operands):
                continue
            device_recorded = scope.access.accesses.get("device", {}).get(path)
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
                for item in _call_accesses(scope, call, "unit", path=path)
            }
            for access in _call_accesses(scope, call, path=path):
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
                mesh_parameters = _mesh_parameters(
                    scope, access.relation.relation, values=access.relation.values
                )
                position = (
                    _wave_position(
                        access.relation.constrain(_first(loops)).relation, mesh_parameters
                    )
                    if wave_units < declared_units and mesh_parameters
                    else None
                )
                wave_stated = not (
                    wave_units < declared_units and mesh_parameters and position is None
                )
                found.append(
                    MovingBoundary(
                        scope=scope,
                        loops=loops,
                        dependency=path[0].expr
                        if (
                            (source_scope := definitions.get(id(access.target.expr))) is not None
                            and source_scope.root.owner is not root.owner
                        )
                        else call.args[access.input_index]
                        if access.input_index is not None
                        else access.target.expr,
                        access=access,
                        space_wave_access=space_wave_access,
                        unit_access=unit_access,
                        dtype=leaves[0].dtype,
                        mesh=mesh,
                        reads=movement.read > 0,
                        wave_units=wave_units,
                        wave_stated=wave_stated,
                        mesh_parameters=mesh_parameters,
                        axis_parameters=axis_parameters
                        if wave_stated
                        and _shared_origin(
                            scope, access.target.expr, definitions, topology_level, wave
                        )
                        else (),
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
        refused = scope.access.refused.get("narrow", frozenset()) | scope.access.refused.get(
            "unit", frozenset()
        )
        for call in refused:
            found.extend(
                (scope, item)
                for item in _uncounted_boundaries(call, whole, AnalysisPrecision.UNKNOWN)
            )
        for path, (call, _recorded) in scope.access.accesses.get("narrow", {}).items():
            if ctx.frame is not None and ctx.callee_frame(call) is not None:
                continue
            if call in refused:
                continue
            moved = get_metadata(call, MemoryMetadata)
            operands = () if moved is None else moved.operands
            accesses = _call_accesses(scope, call, path=path)
            if len(operands) != len(call.args) + 1:
                found.extend((scope, item) for item in _uncounted_boundaries(call, whole))
                continue
            _origin, loops = _access_scopes(scope, path, ctx)
            if _one_time_fill(loops, operands):
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
                    call, _call_accesses(scope, call, "unit", path=path), operands, whole
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
    path=None,
    loops=None,
    window: int,
) -> tuple[ReachedAddresses, ...] | None:
    """Return one Call's cache-backed addresses, or ``None`` for no stated wave."""
    refused = call in scope.access.refused.get("narrow", ())
    if len(operands) != len(call.args) + 1:
        return _uncounted_boundaries(
            call,
            ctx,
            AnalysisPrecision.UNKNOWN if refused else AnalysisPrecision.LOWER_BOUND,
        )
    if loops is None:
        _origin, loops = _access_scopes(scope, path or (SSAValue(call),), ctx)
    if _one_time_fill(loops, operands):
        return ()

    accesses = _call_accesses(scope, call, path=path)
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

        mesh_parameters = _mesh_parameters(
            scope, access.relation.relation, values=access.relation.values
        )
        position = (
            _wave_position(access.relation.constrain(_first(loops)).relation, mesh_parameters)
            if wave_units < declared_units and mesh_parameters
            else None
        )
        if wave_units < declared_units and mesh_parameters and position is None:
            return None
        relation = (
            access.relation.constrain(_first(loops))
            .project(tuple(s.owner.induction_var for s in loops))
            .relation
        )
        if position is not None:
            relation = _restrict_to_wave(relation, position, wave_units)
        result.append(
            _reached_addresses(
                access,
                relation,
                leaves[0].dtype,
                mesh_parameters,
                precision=AnalysisPrecision.UNKNOWN if refused else AnalysisPrecision.EXACT,
                values=access.relation.values,
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
            or len(candidate.loop_scopes) < len(current.time_scope.loop_scopes)
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


def _window_footprints(boundaries, uncounted, buffers):
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
                    boundary.view("relation", window),
                    boundary.dtype,
                    boundary.mesh_parameters,
                    precision=AnalysisPrecision.EXACT,
                    values=boundary.access.relation.values,
                    ownership=boundary.ownership,
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
    prepared = {
        scope: (
            {
                name: sum(spread.logical for _level, spread in held.kinds)
                for name, held in counted.buffers
            },
            buffer_names(counted.reached),
            counted.precision,
        )
        for scope, counted in footprints.items()
        if counted is not None
    }
    rows: list[ReuseWindow] = []
    for item in buffers.values():
        if item.time_scope not in prepared:
            continue
        amounts, labels, precision = prepared[item.time_scope]
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
                precision=precision,
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


def _count(item):
    if item.reached is None:
        return 0, item.precision
    image = item.reached
    mesh_params = tuple(
        name
        for name, value in item.values.items()
        if isinstance(value, Call) and isinstance(value.target, MeshCoord)
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
                held = leaves_of(item.target.expr.type)[item.target.index]
                value = widest_allowed(isl.map.from_range(image), name, held)
                if value is not None:
                    image = image.fix_val(isl.dim_type.PARAM, axis, isl.val(value))
                bounded = bounded.join(AnalysisPrecision.UPPER_BOUND)
            image = image.project_out(isl.dim_type.PARAM, axis, 1)
    image = _project_mesh_parameters(image, tuple((name, None) for name in mesh_params))
    amount = cardinality(image)
    if amount is None:
        return 0, (
            AnalysisPrecision.UNKNOWN
            if item.precision is AnalysisPrecision.UNKNOWN
            else AnalysisPrecision.LOWER_BOUND
        )
    if item.may_coordinates is not None and not item.must.is_equal(item.reached):
        must = cardinality(item.must, union=mesh_params)
        if must is not None:
            amount = min(amount, must + item.may_coordinates)
    return -(-(amount * item.dtype.bit_width) // 8), bounded


def counted_precision(items):
    precision = AnalysisPrecision.EXACT
    for item in items:
        precision = precision.join(item.counted[1])
    return precision


def counted_buffers(footprint):
    """Union by SSA identity and instance ownership before packing bytes."""
    groups, items = footprint._grouped, footprint.reached
    labels = buffer_names(items)
    totals = {}
    for windows, group in groups:
        for item in group:
            key = item.target, item.ownership
            totals[key] = totals.get(key, 0) + item.counted[0] * windows
    logical, counted = {}, {}
    for item in items:
        if item.reached is not None:
            key = item.target, item.ownership
            logical[key] = logical.get(key, 0) + item.counted[0]
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


def reuse_windows(root, *, memory_level, wave, ctx, capacity, labels):
    boundaries = moving_boundaries(
        root, memory_level=memory_level, wave=wave, ctx=ctx, labels=labels
    )
    buffers = _by_buffer(boundaries, wave)
    footprints = _window_footprints(
        boundaries,
        _uncounted_movements(root, memory_level=memory_level, ctx=ctx),
        buffers,
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


def buffer_names(items):
    """Assign display names only at the report boundary."""
    values = tuple(dict.fromkeys(item.target for item in items))
    return dict(zip(values, value_labels(value.expr for value in values), strict=True))
