"""Unique addresses one wave touches in a cache-backed level.

The requested enclosing loops are held at their first iteration, the addresses
every unit of one wave reaches are unioned, and both loads and stores occupy the
level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import isl

from tilefoundry import target, utils
from tilefoundry.interfaces import access, cost
from tilefoundry.ir import core, dim, expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple

from .access import Access
from .iteration_scope import IterationScope
from .loop_domain import induction_name
from .metadata import MemoryMetadata, ReuseWindow
from .precision import Precision
from .reach import Footprint, ReachedAddresses, merged, project_mesh_parameters
from .visitor import AnalyzeContext


@dataclass(frozen=True)
class ReuseAxes:
    """Where a second read of one boundary's data can come from.

    ``time`` is the outermost loop dimension whose iterations reach the same
    addresses; ``space`` names the mesh coordinates the addresses do not vary
    with, and ``space_units`` counts the units mapping onto those addresses.
    ``partial`` names axes with overlapping unit images but no modelled savings.
    """

    time: int | None = None
    space: tuple[str, ...] = ()
    space_units: int = 1
    partial: tuple[str, ...] = ()

    @property
    def window(self) -> int | None:
        """Return the loop dimension held for this reuse window.

        Spatial or partial reuse uses ``-1`` for the innermost loop;
        the caller resolves it against the boundary's scope depth.
        """
        if self.time is not None:
            return self.time
        return -1 if self.space or self.partial else None


def static_axes(mesh: types.Mesh) -> "tuple[int, tuple[int, ...], tuple[int, ...]] | None":
    """A mesh's device-numbered positions, when every one of them is a number."""
    try:
        axes = types.flatten(mesh.layout)
        offset = mesh.layout.offset if isinstance(mesh.layout, types.ComposedLayout) else 0
    except ValueError:
        return None
    shape = tuple(expr.static_dim_value(extent) for extent in types.flatten(axes.shape))
    strides = tuple(expr.static_dim_value(stride) for stride in types.flatten(axes.strides))
    if any(value is None for value in (*shape, *strides, expr.static_dim_value(offset))):
        return None
    return offset, shape, strides


def mesh_parameters_of(
    scope: IterationScope, relation: isl.map | None = None, *, values=None
) -> tuple[tuple[str, expr.Call], ...]:
    """Mesh coordinates involved in this boundary, excluding unrelated retained facts."""
    return tuple(
        (name, value)
        for name, value in (scope.domain.values if values is None else values).items()
        if isinstance(value, expr.Call)
        and isinstance(value.target, dim.MeshCoord)
        and (relation is None or relation.find_dim_by_name(isl.dim_type.PARAM, name) >= 0)
    )


def wave_position(relation, parameters):
    image = relation.range()
    mesh_names = {name for name, _ in parameters}
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
    return linear_position(involved)


def linear_position(
    parameters: tuple[tuple[str, expr.Call], ...],
) -> tuple[int, tuple[tuple[str, int], ...]] | None:
    """The layout-image position named by mesh-coordinate parameters."""
    if not parameters:
        return 0, ()
    mesh = parameters[0][1].target.mesh
    image = static_axes(mesh)
    if image is None:
        return None
    offset, shape, strides = image
    terms: list[tuple[str, int]] = []
    for name, coordinate in parameters:
        if coordinate.target.mesh != mesh or not coordinate.args:
            return None
        axis = expr.static_dim_value(coordinate.args[0])
        if axis is None or not 0 <= axis < len(shape):
            return None
        terms.append((name, strides[axis]))
    return offset, tuple(terms)


def restrict_to_wave(
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


def reached_addresses(
    acc: Access,
    relation: isl.map,
    dtype: expr.DType,
    mesh_parameters: tuple[tuple[str, expr.Call], ...],
    *,
    precision: Precision,
    values: Mapping[str, object] | None = None,
    ownership: tuple[str, ...] = (),
) -> ReachedAddresses:
    """Keep the possible image and the boundary's counting evidence together."""
    reached = relation.range()
    must = reached
    may_coordinates = 0
    if acc.may_access:
        must = isl.set.empty(reached.get_space())
        may_coordinates = utils.cardinality(
            relation.domain(), union=tuple(name for name, _ in mesh_parameters)
        )
        if may_coordinates is None:
            may_coordinates = utils.cardinality(
                reached, union=tuple(name for name, _ in mesh_parameters)
            )
    return ReachedAddresses(
        target=acc.target,
        ownership=ownership,
        output_index=acc.output_index,
        dtype=dtype,
        reached=reached,
        precision=precision.join(Precision.UPPER_BOUND)
        if acc.relation.may_access and not acc.may_access
        else precision,
        must=must,
        may_coordinates=may_coordinates,
        values=acc.relation.aligned(isl.map.from_range(reached), values or {}).values,
    )


@dataclass
class MovingBoundary:
    """One moving boundary at this level, with reached addresses cached by window."""

    scope: IterationScope
    dependency: expr.Expr
    access: Access
    space_wave_access: Access
    unit_access: Access
    dtype: expr.DType
    mesh: types.Mesh | None
    reads: bool
    wave_units: int
    wave_stated: bool
    mesh_parameters: tuple[tuple[str, expr.Call], ...]
    axis_parameters: tuple[str | None, ...]
    position: tuple[int, tuple[tuple[str, int], ...]] | None
    ownership: tuple[str, ...] = ()
    loops: tuple[IterationScope, ...] = ()

    views: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def depth(self):
        return len(self.loop_scopes)

    @property
    def loop_scopes(self):
        return self.scope.loop_scopes

    def view(self, kind, window, axis=None):
        key = kind, window, axis
        if key in self.views:
            return self.views[key]
        acc = {
            "held": self.access,
            "wave": self.space_wave_access,
            "unit": self.unit_access,
            "relation": self.access,
        }[kind]
        loops = self.loop_scopes
        relation = acc.relation.constrain(first_iteration(loops[: window + 1]))
        relation = relation.project(tuple(s.owner.induction_var for s in self.loops or loops))
        if self.position is not None:
            relation = relation.aligned(
                restrict_to_wave(relation.relation, self.position, self.wave_units)
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
                    self.views[key] = None
                    return None
                relation = relation.aligned(
                    relation.relation.fix_val(isl.dim_type.PARAM, index, first)
                )
        result = (
            relation.relation
            if kind == "relation"
            else project_mesh_parameters(relation.relation.range(), self.mesh_parameters)
        )
        self.views[key] = result
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


def space_axes(
    boundary: MovingBoundary, wave: tuple[int, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return mesh axes where one unit reaches the wave-unioned addresses."""
    if boundary.mesh is None or wave[0] <= 1:
        return (), ()
    window = boundary.depth - 1
    wave_reached = boundary.view("wave", window)
    shared, partial = [], []
    reached = boundary.view("relation", -1).range()
    shape = types.flatten(types.flatten(boundary.mesh.layout).shape)
    for axis, parameter_name in enumerate(boundary.axis_parameters):
        extent = expr.static_dim_value(shape[axis])
        unit_reached = boundary.view("unit", window, parameter_name)
        if extent is not None and extent > 1 and unit_reached is not None:
            if wave_reached.is_equal(unit_reached):
                shared.append(axis)
        if parameter_name is None:
            continue
        parameter = reached.find_dim_by_name(isl.dim_type.PARAM, parameter_name)
        if parameter < 0:
            continue
        by_unit = isl.map.from_range(reached).move_dims(
            isl.dim_type.IN, 0, isl.dim_type.PARAM, parameter, 1
        )
        deltas = by_unit.apply_range(by_unit.reverse()).deltas()
        if not deltas.is_subset(deltas.fix_val(isl.dim_type.SET, 0, 0)):
            partial.append(axis)
    return tuple(shared), tuple(partial)


def axis_label(mesh: types.Mesh, axis: int) -> str:
    """Name a mesh axis, falling back to its factual numeric position."""
    stated = mesh.layout.outer if isinstance(mesh.layout, types.ComposedLayout) else mesh.layout
    levels = types.flatten(
        tuple(
            int_tuple.repeat_like(mode, getattr(level, "name", str(level)))
            for mode, level in zip(stated.shape, mesh.topologies, strict=True)
        )
    )
    level_name = levels[axis] if axis < len(levels) else ""
    if axis < len(mesh.names):
        name = mesh.names[axis]
        return f"{level_name}.{name}" if level_name else name
    return f"{level_name}[{axis}]" if level_name else f"[{axis}]"


def shared_units(mesh: types.Mesh, axes: tuple[int, ...], wave: tuple[int, int]) -> int:
    """Count wave positions after projecting out axes that do not share data."""
    image = static_axes(mesh)
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
    units = utils.cardinality(points)
    return units if units is not None and units > 1 else 1


def reuse_axes(boundary: MovingBoundary, wave: tuple[int, int]) -> ReuseAxes:
    """Name reuse axes and how many units share this boundary's addresses."""
    time = time_axis(boundary)
    axes, partial = space_axes(boundary, wave)
    units = shared_units(boundary.mesh, axes, wave) if boundary.mesh and axes else 1
    if units <= 1:
        axes = ()
    partial = partial if time is None and not axes else ()
    return ReuseAxes(
        partial=tuple(axis_label(boundary.mesh, axis) for axis in partial),
        time=time,
        space=tuple(axis_label(boundary.mesh, axis) for axis in axes) if boundary.mesh else (),
        space_units=units,
    )


def boundary_type(acc: Access, call: expr.Call, ctx: cost.Context) -> types.Type | None:
    """Recover the source Type for one input or indexed output boundary."""
    held = ctx.local_type_of(acc.target.expr)
    fields = ir_utils.leaves_of(held)
    return fields[acc.target.index] if 0 <= acc.target.index < len(fields) else None


def call_accesses(
    scope: IterationScope, call: expr.Call, view: str = "narrow", *, path=None
) -> tuple[Access, ...]:
    """The input and output boundaries recorded for one Call in a view."""
    found: list[Access] = []
    for side in (scope.access.accesses.get(view, {}), scope.access.outputs.get(view, {})):
        recorded = side.get(path or (core.SSAValue(call),))
        if recorded is not None and recorded[0] is call:
            found.extend(recorded[1])
    return tuple(found)


def missing_boundaries(
    call: expr.Call,
    accesses: tuple[Access, ...],
    operands: tuple[cost.TrafficBytes, ...],
    ctx: cost.Context,
) -> tuple[ReachedAddresses, ...]:
    """Represent boundary-index gaps as uncounted lower-bound evidence."""
    recorded_inputs = {acc.input_index for acc in accesses if acc.input_index is not None}
    recorded_outputs = {
        acc.output_index for acc in accesses if acc.output_index is not None
    }
    try:
        output = ctx.local_type_of(call)
    except ValueError:
        return (ReachedAddresses(core.SSAValue(call, 0), None, None, None, Precision.LOWER_BOUND),)
    output_count = len(output.fields) if isinstance(output, types.TupleType) else 1
    missing_inputs = (
        ReachedAddresses(
            core.SSAValue(call.args[index], 0), None, None, None, Precision.LOWER_BOUND
        )
        for index in sorted(set(range(len(call.args))) - recorded_inputs)
        if operands[index].read > 0 or operands[index].write > 0
    )
    missing_outputs = (
        ReachedAddresses(core.SSAValue(call, index), index, None, None, Precision.LOWER_BOUND)
        for index in sorted(set(range(output_count)) - recorded_outputs)
        if operands[-1].read > 0 or operands[-1].write > 0
    )
    return (*missing_inputs, *missing_outputs)


def uncounted_boundaries(
    call: expr.Call,
    ctx: cost.Context,
    precision: Precision = Precision.LOWER_BOUND,
) -> tuple[ReachedAddresses, ...]:
    """Represent every boundary when no positional movement answer exists."""
    try:
        output = ctx.local_type_of(call)
    except ValueError:
        return (ReachedAddresses(core.SSAValue(call, 0), None, None, None, precision),)
    output_count = len(output.fields) if isinstance(output, types.TupleType) else 1
    return (
        *(ReachedAddresses(core.SSAValue(arg, 0), None, None, None, precision) for arg in call.args),
        *(
            ReachedAddresses(core.SSAValue(call, index), index, None, None, precision)
            for index in range(output_count)
        ),
    )


def axis_parameters_of(scope: IterationScope, mesh: types.Mesh | None) -> tuple[str | None, ...]:
    """Map each wave-mesh axis to its retained isl parameter."""
    if mesh is None:
        return ()
    names_by_axis = {}
    values = {
        name: value
        for records in (scope.access.accesses, scope.access.outputs)
        for _, accesses in records.get("device", {}).values()
        for acc in accesses
        for name, value in acc.relation.values.items()
    }
    for name, coordinate in mesh_parameters_of(scope, values=values):
        axis = expr.static_dim_value(coordinate.args[0])
        if axis is None:
            continue
        start = 0
        for level in types.separate(coordinate.target.mesh):
            rank = len(types.flatten(types.flatten(level.layout).shape))
            if level == mesh and start <= axis < start + rank:
                names_by_axis[axis - start] = name
            start += rank
    return tuple(
        names_by_axis.get(axis) for axis in range(len(types.flatten(types.flatten(mesh.layout).shape)))
    )


def wave_mesh(scope: IterationScope, topology_level: str) -> types.Mesh | None:
    """Find the enclosing mesh, including the definition's checked outer mesh."""
    for ctx in scope.ancestors():
        mesh = (
            ctx.owner.mesh
            if isinstance(ctx.owner, core.MeshRegion)
            else (ctx.mesh if ctx.parent is None else None)
        )
        if mesh is not None:
            for component in types.separate(mesh):
                topology = component.topologies[0]
                if getattr(topology, "name", topology) == topology_level:
                    return component
    return None


def first_iteration(loops):
    result = access.Relation(isl.map("{ [] -> [] }"))
    for scope in loops:
        result = result.constrain(scope.first((scope,)))
    return result


def one_time_fill(loops, operands: tuple[cost.TrafficBytes, ...]) -> bool:
    """A write-only initializer outside loops streams once, without cache reuse."""
    return not loops and all(item.read == 0 for item in operands)


def shared_origin(scope, tgt, definitions, topology_level, wave) -> bool:
    """A material definition made by multiple wave units is not shared data.

    Registered storage-sharing edges have already resolved the origin, so a
    view of an enclosing definition does not become fresh within this mesh.
    """
    origin = definitions.get(
        id(tgt),
        scope
        if isinstance(tgt, expr.Call) and core.SSAValue(tgt) not in scope.provenance.snapshot
        else None,
    )
    mesh = None if origin is None else wave_mesh(origin, topology_level)
    if mesh is None:
        return True
    axes = tuple(range(len(types.flatten(types.flatten(mesh.layout).shape))))
    return shared_units(mesh, axes, wave) <= 1


def moving_boundaries(
    root: IterationScope,
    *,
    memory_level: str,
    wave: tuple[int, int],
    ctx: AnalyzeContext,
) -> tuple[MovingBoundary, ...]:
    """Collect boundaries that move bytes at *memory_level* in one scope walk.

    Definition-time source entries already name each boundary's material result.
    """
    whole = replace(ctx, topology_level=None, topologies=())
    wave_units, declared_units = wave
    topology_level = ctx.topology_level or ctx.target.get_facts(target.TopologyFacts).parallel_level
    definitions = {
        value_id: scope
        for definition in ctx.definitions.values()
        for value_id, scope in definition.snapshot.definition.locations.items()
        if value_id in definition.access.relations
        or any(
            id(phi) == value_id
            and scope.provenance.lookup(core.SSAValue(initial)).root
            != scope.provenance.lookup(core.SSAValue(yielded)).root
            for phi, initial, yielded in scope.owner.carries()
        )
    }
    found: list[MovingBoundary] = []
    for path, (call, _) in root.access.accesses.get("narrow", {}).items():
        scope = root.at(path[:1])
        mesh = wave_mesh(scope, topology_level)
        axis_parameters = axis_parameters_of(scope, mesh)
        _, loops = scope.access_environment(path)
        moved = expr.get_metadata(call, MemoryMetadata)
        if moved is None or len(moved.operands) != len(call.args) + 1:
            continue
        if one_time_fill(loops, moved.operands):
            continue
        device_recorded = root.access.accesses.get("device", {}).get(path)
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
            for item in call_accesses(scope, call, "unit", path=path)
        }
        for acc in call_accesses(scope, call, path=path):
            movement = (
                moved.operands[acc.input_index]
                if acc.input_index is not None
                else moved.operands[-1]
            )
            if movement.read <= 0 and movement.write <= 0:
                continue
            held = boundary_type(acc, call, whole)
            if held is None:
                continue
            leaves = ir_utils.leaves_of(held)
            if memory_level not in {str(leaf.storage) for leaf in leaves}:
                continue
            if len(leaves) != 1 or not isinstance(leaves[0], types.TensorType):
                continue
            boundary_key = (acc.input_index, acc.output_index, acc.target.index)
            space_wave_access = device_by_boundary.get(boundary_key, acc)
            unit_access = unit_by_boundary.get(boundary_key)
            if unit_access is None:
                continue
            mesh_parameters = mesh_parameters_of(
                scope, acc.relation.relation, values=acc.relation.values
            )
            position = (
                wave_position(
                    acc.relation.constrain(first_iteration(loops)).relation, mesh_parameters
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
                        (source_scope := definitions.get(id(acc.target.expr))) is not None
                        and source_scope.root.owner is not root.owner
                    )
                    else call.args[acc.input_index]
                    if acc.input_index is not None
                    else acc.target.expr,
                    access=acc,
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
                    and shared_origin(scope, acc.target.expr, definitions, topology_level, wave)
                    else (),
                    position=position,
                )
            )
    return tuple(found)


def uncounted_movements(
    root: IterationScope,
    *,
    memory_level: str,
    ctx: cost.Context,
) -> tuple[tuple[IterationScope, ReachedAddresses], ...]:
    """Keep completeness evidence that has no countable moving boundary."""
    whole = replace(ctx, topology_level=None, topologies=())
    found: list[tuple[IterationScope, ReachedAddresses]] = []
    for path, (call, _) in root.access.accesses.get("narrow", {}).items():
        scope = root.at(path[:1])
        refused = scope.access.refused.get("narrow", frozenset()) | scope.access.refused.get("unit", frozenset())
        if call in refused:
            found.extend(
                (scope, item) for item in uncounted_boundaries(call, whole, Precision.UNKNOWN)
            )
            continue
        if ctx.module is not None and ctx.callee_context(call) is not None:
            continue
        if call in refused:
            continue
        moved = expr.get_metadata(call, MemoryMetadata)
        operands = () if moved is None else moved.operands
        accesses = call_accesses(scope, call, path=path)
        if len(operands) != len(call.args) + 1:
            found.extend((scope, item) for item in uncounted_boundaries(call, whole))
            continue
        _, loops = scope.access_environment(path)
        if one_time_fill(loops, operands):
            continue
        for acc in accesses:
            movement = (
                operands[acc.input_index] if acc.input_index is not None else operands[-1]
            )
            if movement.read <= 0 and movement.write <= 0:
                continue
            held = boundary_type(acc, call, whole)
            if held is None:
                continue
            leaves = ir_utils.leaves_of(held)
            if memory_level in {str(leaf.storage) for leaf in leaves} and (
                len(leaves) != 1 or not isinstance(leaves[0], types.TensorType)
            ):
                found.append(
                    (
                        scope,
                        ReachedAddresses(
                            acc.target,
                            acc.output_index,
                            None,
                            None,
                            Precision.LOWER_BOUND,
                        ),
                    )
                )
        found.extend(
            (scope, item)
            for item in missing_boundaries(
                call, call_accesses(scope, call, "unit", path=path), operands, whole
            )
        )
    return tuple(found)


def reached_by(
    scope: IterationScope,
    call: expr.Call,
    *,
    memory_level: str,
    wave_units: int,
    declared_units: int,
    operands: tuple[cost.TrafficBytes, ...],
    ctx: cost.Context,
    path=None,
    loops=None,
    window: int,
) -> tuple[ReachedAddresses, ...] | None:
    """Return one Call's cache-backed addresses, or ``None`` for no stated wave."""
    refused = call in scope.access.refused.get("narrow", ())
    if len(operands) != len(call.args) + 1:
        return uncounted_boundaries(
            call,
            ctx,
            Precision.UNKNOWN if refused else Precision.LOWER_BOUND,
        )
    if loops is None:
        _, loops = scope.access_environment(path or (core.SSAValue(call),))
    if one_time_fill(loops, operands):
        return ()

    accesses = call_accesses(scope, call, path=path)
    result: list[ReachedAddresses] = []
    for acc in accesses:
        moved = operands[acc.input_index] if acc.input_index is not None else operands[-1]
        if moved.read <= 0 and moved.write <= 0:
            continue
        held = boundary_type(acc, call, ctx)
        if held is None:
            continue
        leaves = ir_utils.leaves_of(held)
        if memory_level not in {str(leaf.storage) for leaf in leaves}:
            continue
        if len(leaves) != 1 or not isinstance(leaves[0], types.TensorType):
            result.append(
                ReachedAddresses(
                    target=acc.target,
                    output_index=acc.output_index,
                    dtype=None,
                    reached=None,
                    precision=Precision.LOWER_BOUND,
                )
            )
            continue

        mesh_parameters = mesh_parameters_of(
            scope, acc.relation.relation, values=acc.relation.values
        )
        position = (
            wave_position(acc.relation.constrain(first_iteration(loops)).relation, mesh_parameters)
            if wave_units < declared_units and mesh_parameters
            else None
        )
        if wave_units < declared_units and mesh_parameters and position is None:
            return None
        relation = (
            acc.relation.constrain(first_iteration(loops))
            .project(tuple(s.owner.induction_var for s in loops))
            .relation
        )
        if position is not None:
            relation = restrict_to_wave(relation, position, wave_units)
        result.append(
            reached_addresses(
                acc,
                relation,
                leaves[0].dtype,
                mesh_parameters,
                precision=Precision.UNKNOWN if refused else Precision.EXACT,
                values=acc.relation.values,
            )
        )
    result.extend(missing_boundaries(call, accesses, operands, ctx))
    return tuple(result)


@dataclass
class BufferReuse:
    """The selected reuse axes for one source buffer."""

    boundary: MovingBoundary
    time_scope: IterationScope | None
    space: tuple[str, ...]
    space_units: int
    partial: tuple[str, ...]


def by_buffer(
    boundaries: tuple[MovingBoundary, ...], wave: tuple[int, int]
) -> dict[tuple[core.SSAValue, tuple[str, ...]], BufferReuse]:
    """Select the outermost time axis and union space axes per buffer."""
    buffers: dict[tuple[core.SSAValue, tuple[str, ...]], BufferReuse] = {}
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
        buffers[key] = BufferReuse(
            boundary=boundary if current is None else current.boundary,
            time_scope=selected,
            space=space,
            space_units=max(axes.space_units, current.space_units if current else 1),
            partial=tuple(dict.fromkeys((*(current.partial if current else ()), *axes.partial))),
        )
    return buffers


def window_footprints(boundaries, uncounted, buffers):
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
                reached_addresses(
                    boundary.access,
                    boundary.view("relation", window),
                    boundary.dtype,
                    boundary.mesh_parameters,
                    precision=Precision.EXACT,
                    values=boundary.access.relation.values,
                    ownership=boundary.ownership,
                )
            )
        if time_scope in footprints:
            continue
        reached.extend(
            item
            for scope, item in uncounted
            if time_scope is None or time_scope in scope.loop_scopes
        )
        footprints[time_scope] = Footprint(((1, merged(reached)),))
    return footprints


def reuse_rows(
    buffers: Mapping[tuple[core.SSAValue, tuple[str, ...]], BufferReuse],
    footprints: Mapping[IterationScope | None, Footprint | None],
    capacity: int,
) -> tuple[ReuseWindow, ...]:
    """Build sorted report rows from selected axes and counted windows."""
    prepared = {
        scope: (
            {
                (value, ownership): sum(spread.logical for _, spread in held.kinds)
                for value, ownership, held in counted.buffers
            },
            counted.precision,
        )
        for scope, counted in footprints.items()
        if counted is not None
    }
    rows: list[ReuseWindow] = []
    for item in buffers.values():
        if item.time_scope not in prepared:
            continue
        amounts, precision = prepared[item.time_scope]
        key = item.boundary.access.target, item.boundary.ownership
        holds = sum(amounts.values())
        repeats = (1 if item.time_scope is None else item.time_scope.trips()) * item.space_units
        reuse = max(0, repeats - 1) * amounts.get(key, 0)
        if reuse == 0:
            if not item.partial:
                continue
            reuse, precision = None, Precision.UNKNOWN
        rows.append(
            ReuseWindow(
                buffer=key[0],
                ownership=key[1],
                time=(
                    induction_name(item.time_scope.owner)
                    if item.time_scope is not None and isinstance(item.time_scope.owner, core.LoopRegion)
                    else ""
                ),
                space=",".join(item.space or item.partial),
                holds_bytes=holds,
                reuse_bytes=reuse,
                fits=holds < capacity,
                precision=precision,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.reuse_bytes or 0, reverse=True))


def reuse_windows(root, *, memory_level, wave, ctx, capacity):
    boundaries = moving_boundaries(root, memory_level=memory_level, wave=wave, ctx=ctx)
    buffers = by_buffer(boundaries, wave)
    footprints = window_footprints(
        boundaries,
        uncounted_movements(root, memory_level=memory_level, ctx=ctx),
        buffers,
    )
    return reuse_rows(buffers, footprints, capacity)


def wave_of(
    module: core.Module,
    tgt: target.Target,
    topology_level: str | None,
    *,
    topologies: tuple[Topology, ...] | None = None,
) -> tuple[int, int] | None:
    """Return ``(wave_units, declared_units)``, or ``None`` when unstated."""
    try:
        facts = tgt.get_facts(target.TopologyFacts)
        level = facts.parallel() if topology_level is None else facts.level(topology_level)
        if level is None or level.max_physical_units is None:
            return None
        declared_units = expr.static_dim_value(
            (
                module.resolve_topology(level.name)
                if topologies is None
                else next(item for item in topologies if item.name == level.name)
            ).size
        )
    except (target.UnsupportedCapabilityError, ValueError):
        return None
    if declared_units is None:
        return None
    return min(declared_units, level.max_physical_units), declared_units


def cached_level(facts: target.MemoryHierarchyFacts) -> tuple[str, str, int] | None:
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
    "ReuseAxes",
    "axis_label",
    "cached_level",
    "moving_boundaries",
    "reached_by",
    "reuse_axes",
    "reuse_windows",
    "shared_units",
    "space_axes",
    "time_axis",
    "wave_of",
]
