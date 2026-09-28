"""HIR operation that selects one registered TIR instruction for a tensor tile."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Tuple

import isl

from tilefoundry.evaluator.registry import register_eval, schedule_eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Op, Var
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.pattern import (
    AndPattern,
    ComposedLayoutPattern,
    DistinctConstraint,
    LayoutPattern,
    OrPattern,
    PatternMatcher,
    ShardLayoutPattern,
    SwitchPattern,
    Tensor,
    TensorPattern,
    between_rules,
    evaluated,
)
from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    UnitType,
)
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import separate, starts
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry import register_cost_evaluator, register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    iteration_universe,
    projected,
    reached_elements,
    register_access_relation,
    relation_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import Cost, TrafficBytes


@register_op(dialect="tf", category="schedule", name="schedule")
class ScheduleOp(Op):
    """Apply one TIR instruction repeatedly over a tuple of HIR operands."""

    operands = ParamDef(kind="input", annotation=Tuple[Tensor], pattern=Tensor)
    op = ParamDef(kind="attribute", annotation=Op)
    repeat = ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    order = ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    buffers = ParamDef(kind="attribute", annotation=int, default=1)

    def __init__(self, **attrs) -> None:
        super().__init__(**attrs)
        if self.repeat is not None and (
            not isinstance(self.repeat, tuple)
            or any(type(value) is not int or value < 1 for value in self.repeat)
        ):
            raise ValueError("schedule repeat must be a tuple of positive integers")
        if self.order is not None and (
            not isinstance(self.order, tuple) or any(type(value) is not int for value in self.order)
        ):
            raise ValueError("schedule order must be a tuple of integer positions")
        if type(self.buffers) is not int or self.buffers < 1:
            raise ValueError("schedule buffers must be a positive integer")


@dataclass(frozen=True)
class IssueAxis:
    """One work axis of a scheduled instruction issue."""

    name: str
    extent: int
    atom: int
    repeat: int
    row_copies: int
    is_group: bool


@dataclass(frozen=True)
class IssuePlan:
    """The lowering facts derived for one scheduled instruction issue.

    ``operand_axes`` follows instruction schema order, then tensor-coordinate
    order; ``None`` marks a coordinate that is not a projection of one work axis.
    """

    repeat: tuple[int, ...]
    order: tuple[int, ...]
    single_shape: tuple[int, ...]
    operand_types: tuple[TensorType, ...]
    axes: tuple[IssueAxis, ...]
    operand_axes: tuple[tuple[str | None, ...], ...]


@register_eval(ScheduleOp)
def _eval_schedule(ctx):
    """Evaluate the SSA value produced by the selected TIR instruction."""
    selected = ctx.op.op
    handler = schedule_eval_registry.lookup(type(selected))
    if handler is None:
        raise EvalError(f"no schedule evaluator registered for {type(selected).__name__}")
    return handler(ctx.for_op(selected, ctx.args, ctx.result_type))


class _RelationContext:
    """The operand types of a synthetic, registry-dispatched instruction call."""

    def __init__(self, types: dict[int, TensorType]) -> None:
        self._types = types

    def type_of(self, expr: Var) -> TensorType:
        return self._types[id(expr)]


def _single_issue_relations(op: Op, operand_types: tuple[TensorType, ...]) -> AccessRelations:
    """Ask the access-relation registry about one synthetic instruction call."""
    args = tuple(
        Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(operand_types)
    )
    call = Call(target=op, args=args, type=UnitType())
    return relations_of(call, _RelationContext(dict(zip(map(id, args), operand_types))))


def _selected(pattern, bindings: dict):
    """Select declaration branches fixed by instruction attributes."""
    if isinstance(pattern, SwitchPattern) and pattern.param in bindings:
        branch = dict(pattern.branches).get(bindings[pattern.param])
        return None if branch is None else _selected(branch, bindings)
    if isinstance(pattern, (AndPattern, OrPattern)):
        parts = pattern.parts if isinstance(pattern, AndPattern) else pattern.patterns
        selected = tuple(held for part in parts if (held := _selected(part, bindings)) is not None)
        if len(selected) == 1:
            return selected[0]
    return pattern


def _operand_pattern(param: ParamDef, op: Op, bindings: dict) -> TensorPattern | None:
    pattern = param.pattern
    if hasattr(pattern, "read_on"):
        pattern = pattern.read_on(op)
    pattern = _selected(pattern, bindings)
    return pattern if isinstance(pattern, TensorPattern) else None


def _fixed(value, bindings: dict):
    if isinstance(value, tuple):
        parts = tuple(_fixed(item, bindings) for item in value)
        return None if any(item is None for item in parts) else parts
    if value is None or type(value) in (int, str):
        return value
    name = getattr(value, "name", None)
    if name in bindings:
        return bindings[name]
    resolved = evaluated(value, bindings)
    return resolved if resolved is not None else value if not hasattr(value, "name") else None


def _declared_shape(pattern: TensorPattern | None, bindings: dict) -> tuple | None:
    if pattern is None or pattern.shape is None:
        return None
    shape = _fixed(pattern.shape, bindings)
    return shape if isinstance(shape, tuple) and all(type(dim) is int for dim in shape) else None


def _row_room(shape: tuple, strides: tuple) -> tuple[int, int] | None:
    if not any(isinstance(mode, tuple) for mode in shape):
        return None
    modes = [
        (axis, extent, step)
        for axis, (group, steps) in enumerate(zip(shape, strides, strict=True))
        for extent, step in zip(flatten(group), flatten(steps), strict=True)
        if extent != 1
    ]
    if any(type(value) is not int for _, extent, step in modes for value in (extent, step)):
        return None
    units = [(axis, extent) for axis, extent, step in modes if step == 1]
    above = [step for _, _, step in modes if step > 1]
    if len(units) != 1 or not above:
        return None
    (axis, extent), next_step = units[0], min(above)
    fastest = [mode for mode in modes if mode[0] == axis][-1]
    if fastest[2] != 1 or next_step <= extent or next_step % extent:
        return None
    return axis, next_step // extent


def _tile_modes(shape: tuple, strides: tuple, counts: tuple[int, ...]) -> tuple[tuple, tuple]:
    room = _row_room(shape, strides)
    if room is not None and counts[room[0]] > 1:
        axis, copies = room
        count = counts[axis]
        if count % copies and copies % count:
            raise ValueError(f"{count} copies along tensor axis {axis} are no whole number of rows")
        beside = min(count, copies)
        extents = tuple(flatten(shape[axis]))
        widened = list(shape)
        widened[axis] = (
            (*extents[:-1], extents[-1] * beside) if len(extents) > 1 else extents[-1] * beside
        )
        shape, strides = _tile_modes(
            tuple(widened),
            strides,
            tuple(count // beside if index == axis else held for index, held in enumerate(counts)),
        )
        shape, strides = list(shape), list(strides)
        shape[axis] = (*flatten(shape[axis])[:-1], beside, extents[-1])
        strides[axis] = (*flatten(strides[axis])[:-1], extents[-1], 1)
        return tuple(shape), tuple(strides)

    step = prod(flatten(shape))
    added = []
    for axis, count in reversed(tuple(enumerate(counts))):
        if count != 1:
            added.insert(0, (axis, count, step))
            step *= count
    if not added:
        return shape, strides
    if not any(isinstance(mode, tuple) for mode in shape):
        return (
            (*(count for _, count, _ in added), *shape),
            (*(step for _, _, step in added), *strides),
        )
    if len(shape) != len(counts):
        raise ValueError(f"{len(shape)} layout axes cannot be repeated {len(counts)} ways")
    held_shape, held_strides = list(shape), list(strides)
    for axis, count, step in added:
        held_shape[axis] = (count, *flatten(held_shape[axis]))
        held_strides[axis] = (step, *flatten(held_strides[axis]))
    return tuple(held_shape), tuple(held_strides)


def _untile_modes(
    shape: tuple, strides: tuple, counts: tuple[int, ...]
) -> tuple[tuple, tuple] | None:
    grouped = any(isinstance(mode, tuple) for mode in shape)
    if not grouped:
        shift = sum(count != 1 for count in counts)
        if shift >= len(shape):
            return None
        fragment = tuple(shape[shift:]), tuple(strides[shift:])
    else:
        if len(shape) != len(counts):
            return None
        held_shape, held_strides = list(shape), list(strides)
        for axis, count in enumerate(counts):
            if count == 1:
                continue
            extents, steps = tuple(flatten(shape[axis])), tuple(flatten(strides[axis]))
            if len(extents) < 2:
                return None
            held_shape[axis] = extents[1:] if len(extents) > 2 else extents[1]
            held_strides[axis] = steps[1:] if len(steps) > 2 else steps[1]
        fragment = tuple(held_shape), tuple(held_strides)
    return fragment if _tile_modes(*fragment, counts) == (shape, strides) else None


def _fragment_mesh(source: Mesh, required: Mesh) -> Mesh:
    required_name = getattr(required.topologies[0], "name", required.topologies[0])
    source_level = next(
        level
        for level in separate(source)
        if getattr(level.topologies[0], "name", level.topologies[0]) == required_name
    )
    start = starts(source_level)[0]
    layout = ComposedLayout(None, start, required.layout)
    return Mesh(source_level.topologies, layout, required.names)


def _single_layout(
    layout,
    counts: tuple[int, ...],
    pattern,
    op: Op,
    consumer_mesh: Mesh | None,
):
    if layout is None:
        return layout
    if isinstance(layout, ShardLayout):
        if all(count == 1 for count in counts):
            return layout
        fragment = _single_layout(layout.layout, counts, None, op, consumer_mesh)
        if fragment is None:
            return None
        declared = pattern
        attrs = layout.attrs
        mesh = layout.mesh
        if isinstance(declared, ShardLayoutPattern):
            attrs = declared.attrs
            required = getattr(getattr(op, "atom", None), "required_scope", None)
            if required is not None:
                mesh = _fragment_mesh(layout.mesh, required)
        return ShardLayout(fragment, attrs, mesh)
    if isinstance(layout, ComposedLayout):
        if layout.inner is not None and not isinstance(layout.inner, Swizzle):
            return None
        fragment = _single_layout(layout.outer, counts, None, op, consumer_mesh)
        result = None if fragment is None else ComposedLayout(layout.inner, 0, fragment)
    elif isinstance(layout, Layout) and layout.strides is not None:
        if all(count == 1 for count in counts):
            result = layout
        else:
            fragment = _untile_modes(tuple(layout.shape), tuple(layout.strides), counts)
            result = None if fragment is None else Layout(*fragment)
    else:
        return None
    if (
        result is not None
        and isinstance(pattern, ShardLayoutPattern)
        and all(isinstance(attr, Broadcast) for attr in pattern.attrs)
        and consumer_mesh is not None
        and (required := getattr(getattr(op, "atom", None), "required_scope", None)) is not None
    ):
        result = ShardLayout(
            result,
            pattern.attrs,
            _fragment_mesh(consumer_mesh, required),
        )
    return result


def _single_type(
    type_: TensorType,
    pattern: TensorPattern | None,
    bindings: dict,
    op: Op,
    consumer_mesh: Mesh | None,
    counts: tuple[int, ...] | None = None,
) -> TensorType:
    shape = _declared_shape(pattern, bindings)
    if shape is None and counts is not None:
        if len(counts) != len(type_.shape):
            raise ValueError(
                f"operand rank {len(type_.shape)} differs from repeat-derived rank {len(counts)}"
            )
        divided = []
        for whole, count in zip(type_.shape, counts, strict=True):
            if type(whole) is not int or whole % count:
                raise ValueError(f"operand extent {whole} is not divisible by repeat {count}")
            divided.append(whole // count)
        shape = tuple(divided)
    shape = shape or tuple(type_.shape)
    if len(shape) != len(type_.shape):
        return TensorType(shape, type_.dtype, None, type_.storage)
    layout_counts = []
    for whole, single in zip(type_.shape, shape, strict=True):
        if type(whole) is not int or type(single) is not int or single < 1 or whole % single:
            layout_counts.append(1)
        else:
            layout_counts.append(whole // single)
    layout_pattern = None if pattern is None else pattern.layout
    layout = _single_layout(type_.layout, tuple(layout_counts), layout_pattern, op, consumer_mesh)
    return TensorType(shape, type_.dtype, layout, type_.storage)


def _as_it_lies(value, type_: TensorType, ctx) -> TensorType:
    """State the arrangement an instruction reads, apart from the address origin."""
    layout = type_.layout
    if isinstance(value, Call) and isinstance(value.target, Slice):
        if isinstance(layout, ComposedLayout):
            layout = layout.outer
        elif layout is None:
            parent = ctx.type_of(value.args[0])
            parent_layout = parent.layout
            if isinstance(parent_layout, ComposedLayout):
                parent_layout = parent_layout.outer
            strides = (
                tuple(parent_layout.strides)
                if isinstance(parent_layout, Layout) and parent_layout.strides is not None
                else tuple(compact_row_major(parent.shape))
            )
            layout = Layout(tuple(type_.shape), strides)
    elif layout is None:
        layout = Layout(tuple(type_.shape), tuple(compact_row_major(type_.shape)))
    return TensorType(tuple(type_.shape), type_.dtype, layout, type_.storage)


def _thread_mesh(mesh: Mesh | None) -> Mesh | None:
    if mesh is None:
        return None
    return next(
        (
            level
            for level in separate(mesh)
            if getattr(level.topologies[0], "name", level.topologies[0]) == "thread"
        ),
        None,
    )


def _materialize_layout(pattern, bindings: dict, mesh: Mesh | None):
    pattern = _selected(pattern, bindings)
    if isinstance(pattern, LayoutPattern):
        shape = _fixed(pattern.shape, bindings)
        strides = _fixed(pattern.strides, bindings)
        return None if shape is None or strides is None else Layout(shape, strides)
    if isinstance(pattern, ComposedLayoutPattern):
        inner = _fixed(pattern.inner, bindings)
        offset = _fixed(pattern.offset, bindings)
        outer = _materialize_layout(pattern.outer, bindings, mesh)
        return None if offset is None or outer is None else ComposedLayout(inner, offset, outer)
    if isinstance(pattern, ShardLayoutPattern):
        held = _materialize_layout(pattern.layout, bindings, mesh)
        actual_mesh = _thread_mesh(mesh)
        if actual_mesh is not None:
            mesh_bindings = {**bindings, "p0": starts(actual_mesh)[0]}
            stated = _materialize_layout(pattern.mesh.layout, mesh_bindings, mesh)
            if stated is not None:
                actual_mesh = Mesh(actual_mesh.topologies, stated, actual_mesh.names)
        return (
            None
            if held is None or actual_mesh is None
            else ShardLayout(held, pattern.attrs, actual_mesh)
        )
    return None


_LAYOUT_STORAGES = {
    "gmem_layout": StorageKind.GMEM,
    "smem_layout": StorageKind.SMEM,
    "rmem_layout": StorageKind.RMEM,
}


def _result_storage(op: Op, param: ParamDef, pattern: TensorPattern, operands: dict) -> StorageKind:
    stated = [
        storage for name, storage in _LAYOUT_STORAGES.items() if getattr(op, name, None) is not None
    ]
    if len(stated) == 1:
        return stated[0]
    if isinstance(pattern.storage, StorageKind):
        return pattern.storage
    for rule in between_rules(type(op)):
        if not isinstance(rule, DistinctConstraint) or rule.field != "storage":
            continue
        if param.name not in (rule.left, rule.right):
            continue
        other = rule.right if param.name == rule.left else rule.left
        source = operands.get(other)
        if source is None:
            continue
        choices = tuple(
            storage for storage in (StorageKind.GMEM, StorageKind.SMEM) if storage != source.storage
        )
        if len(choices) == 1:
            return choices[0]
    raise ValueError(f"{type(op).__name__} does not determine {param.name} storage")


def _result_type(
    op: Op,
    param: ParamDef,
    pattern: TensorPattern,
    operands: dict[str, TensorType],
    bindings: dict,
    mesh: Mesh | None,
) -> TensorType:
    source = next(iter(operands.values()))
    shape = _declared_shape(pattern, bindings) or tuple(source.shape)
    dtype = pattern.dtype if hasattr(pattern.dtype, "bit_width") else source.dtype
    storage = _result_storage(op, param, pattern, operands)
    layout = next(
        (
            getattr(op, name)
            for name, held_storage in _LAYOUT_STORAGES.items()
            if held_storage is storage and getattr(op, name, None) is not None
        ),
        None,
    )
    if layout is None and pattern.layout is not None:
        layout = _materialize_layout(pattern.layout, bindings, mesh)
    if layout is None:
        layout = Layout(shape, tuple(compact_row_major(shape)))
    return TensorType(shape, dtype, layout, storage)


def _iteration_shape(relations: AccessRelations) -> tuple:
    space = iteration_universe(relations)
    if space is None:
        raise ValueError("instruction access relations state no iteration space")
    return tuple(
        int(space.dim_max(axis).max_val().num_si()) + 1 for axis in range(space.tuple_dim())
    )


def _instruction_schema(
    op: Op,
) -> tuple[tuple[ParamDef, ...], tuple[ParamDef, ...], tuple[ParamDef, ...]]:
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        raise ValueError(f"{type(op).__name__} is not a registered operation")
    params = tuple(param for param in schema.signature if param.kind == "input")
    if any(param.effect is None for param in params):
        raise ValueError(f"{type(op).__name__} does not declare every operand's memory effect")
    reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
    writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    if not writes:
        raise ValueError(f"{type(op).__name__} writes no result operand")
    return params, reads, writes


def _relation_types(
    call: Call, ctx
) -> tuple[
    Op,
    tuple[ParamDef, ...],
    tuple[TensorPattern | None, ...],
    dict,
    tuple[TensorType, ...],
    Mesh | None,
]:
    """Build whole boundaries and selected patterns without reading the Call Type."""
    schedule = call.target
    op = schedule.op
    params, reads, writes = _instruction_schema(op)
    if len(call.args) != len(reads):
        raise ValueError(f"{type(op).__name__} reads {len(reads)} operands, got {len(call.args)}")
    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    patterns = {param.name: _operand_pattern(param, op, bindings) for param in params}
    whole = {param.name: ctx.type_of(arg) for param, arg in zip(reads, call.args, strict=True)}
    for param in writes:
        if param.effect & MemoryEffect.READ:
            continue
        pattern = patterns[param.name]
        if pattern is None:
            raise ValueError(f"{type(op).__name__} {param.name} has no tensor pattern")
        whole[param.name] = _result_type(
            op,
            param,
            pattern,
            whole,
            bindings,
            getattr(ctx, "current_mesh", None),
        )
    mesh = getattr(ctx, "current_mesh", None)
    return (
        op,
        params,
        tuple(patterns[param.name] for param in params),
        bindings,
        tuple(whole[param.name] for param in params),
        mesh,
    )


def _repeat_order(
    schedule: ScheduleOp,
    whole_shape: tuple[int, ...],
    single_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(whole_shape) != len(single_shape):
        raise ValueError("single-issue and scheduled iteration ranks differ")
    repeat = []
    for whole_extent, single_extent in zip(whole_shape, single_shape, strict=True):
        if single_extent < 1 or whole_extent % single_extent:
            raise ValueError(
                f"iteration extent {whole_extent} is not divisible by "
                f"single-issue extent {single_extent}"
            )
        repeat.append(whole_extent // single_extent)
    inferred = tuple(repeat)
    actual = inferred if schedule.repeat is None else schedule.repeat
    if actual != inferred:
        raise ValueError(f"repeat {actual} conflicts with inferred repeat {inferred}")
    return actual, _issue_order(schedule, len(inferred)), single_shape


def _issue_order(schedule: ScheduleOp, rank: int) -> tuple[int, ...]:
    order = tuple(range(rank)) if schedule.order is None else schedule.order
    if sorted(order) != list(range(rank)):
        raise ValueError(f"order {order} is not a permutation of iteration dimensions")
    return order


def _open_repeat_order(schedule: ScheduleOp, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    repeat = (1,) * rank if schedule.repeat is None else schedule.repeat
    if len(repeat) != rank:
        raise ValueError(f"repeat {repeat} has rank {len(repeat)}, expected {rank}")
    if any(count != 1 for count in repeat):
        raise ValueError("transfer tiling is not yet supported")
    return repeat, _issue_order(schedule, rank)


def _axis_names(op: Op, rank: int) -> tuple[str, ...]:
    if getattr(op, "atom", None) is not None and rank == 3:
        return ("m", "n", "k")
    return tuple(f"d{index}" for index in range(rank))


def _operand_axes(
    relations: AccessRelations, axis_names: tuple[str, ...]
) -> tuple[tuple[str | None, ...], ...]:
    """Project each operand coordinate onto at most one work axis."""
    universe = iteration_universe(relations)
    if universe is None:
        raise ValueError("instruction access relations state no iteration space")
    domain_names = tuple(
        universe.get_dim_name(isl.dim_type.SET, index) or f"d{index}"
        for index in range(universe.dim(isl.dim_type.SET))
    )
    aliases = dict(zip(domain_names, axis_names, strict=True))
    return tuple(
        tuple(
            next(
                (
                    aliases.get(domain_names[source_axis])
                    for source_axis in range(relation.dim(isl.dim_type.IN))
                    if _is_projection(relation, source_axis, operand_axis)
                ),
                None,
            )
            for operand_axis in range(relation.dim(isl.dim_type.OUT))
        )
        for relation in (relation_of(boundary.pattern) for boundary in relations.inputs)
    )


def _operand_counts(
    operand_axes: tuple[tuple[str | None, ...], ...],
    axis_names: tuple[str, ...],
    repeat: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    counts = dict(zip(axis_names, repeat, strict=True))
    return tuple(
        tuple(1 if axis is None else counts[axis] for axis in axes) for axes in operand_axes
    )


def _derive_issue(
    schedule: ScheduleOp,
    op: Op,
    patterns: tuple[TensorPattern | None, ...],
    whole_types: tuple[TensorType, ...],
    bindings: dict,
    mesh: Mesh | None,
) -> tuple[IssuePlan, AccessRelations]:
    """Derive fixed-shape and author-directed issues through one path."""
    whole = _single_issue_relations(op, whole_types)
    whole_shape = _iteration_shape(whole)
    axis_names = _axis_names(op, len(whole_shape))
    operand_axes = _operand_axes(whole, axis_names)
    fixed_shape = any(_declared_shape(pattern, bindings) is not None for pattern in patterns)

    if fixed_shape:
        single_types = tuple(
            _single_type(type_, pattern, bindings, op, _thread_mesh(mesh))
            for type_, pattern in zip(whole_types, patterns, strict=True)
        )
        single = _single_issue_relations(op, single_types)
        repeat, order, single_shape = _repeat_order(
            schedule, whole_shape, _iteration_shape(single)
        )
    else:
        repeat, order = _open_repeat_order(schedule, len(whole_shape))
        counts = _operand_counts(operand_axes, axis_names, repeat)
        single_types = tuple(
            _single_type(type_, pattern, bindings, op, _thread_mesh(mesh), held_counts)
            for type_, pattern, held_counts in zip(
                whole_types, patterns, counts, strict=True
            )
        )
        single = _single_issue_relations(op, single_types)
        single_shape = _iteration_shape(single)
        if len(whole_shape) != len(single_shape):
            raise ValueError("single-issue and scheduled iteration ranks differ")

    axes = _issue_axes(op, whole_shape, single_shape, repeat, single_types, operand_axes)
    return IssuePlan(repeat, order, single_shape, single_types, axes, operand_axes), single


def _issue_plan(call: Call, ctx) -> tuple[IssuePlan, AccessRelations]:
    """Derive one plan together with the relations used to build it."""
    op, _params, patterns, bindings, whole_types, mesh = _relation_types(call, ctx)
    return _derive_issue(call.target, op, patterns, whole_types, bindings, mesh)


def _issue_axes(
    op: Op,
    whole_shape: tuple[int, ...],
    single_shape: tuple[int, ...],
    repeat: tuple[int, ...],
    single_types: tuple[TensorType, ...],
    operand_axes: tuple[tuple[str | None, ...], ...],
) -> tuple[IssueAxis, ...]:
    """Derive grouping and row-wise issue facts from the same single issue."""
    atom = getattr(op, "atom", None)
    names = _axis_names(op, len(single_shape))
    copies = {name: 1 for name in names}
    if atom is not None:
        for type_, mapped_axes in zip(single_types, operand_axes, strict=True):
            layout = type_.layout
            if isinstance(layout, ShardLayout):
                layout = layout.layout
            if not (
                isinstance(layout, ComposedLayout)
                and isinstance(layout.inner, Swizzle)
                and isinstance(layout.outer, Layout)
                and layout.outer.strides is not None
            ):
                continue
            room = _row_room(tuple(layout.outer.shape), tuple(layout.outer.strides))
            if room is None:
                continue
            operand_axis, available = room
            axis_name = mapped_axes[operand_axis]
            if axis_name is not None:
                copies[axis_name] = max(copies[axis_name], available)
    return tuple(
        IssueAxis(
            name=name,
            extent=extent,
            atom=atom_extent,
            repeat=count,
            row_copies=copies[name],
            is_group=atom is not None and index == 0 and count > 1,
        )
        for index, (name, extent, atom_extent, count) in enumerate(
            zip(names, whole_shape, single_shape, repeat, strict=True)
        )
    )


def _is_projection(relation: isl.map, source_axis: int, target_axis: int) -> bool:
    """Whether one operand coordinate is exactly one work coordinate."""
    local = isl.local_space.from_space(relation.get_space())
    equal = isl.constraint.alloc_equality(local)
    equal = equal.set_coefficient_si(isl.dim_type.IN, source_axis, 1)
    equal = equal.set_coefficient_si(isl.dim_type.OUT, target_axis, -1)
    projected = isl.map.universe(relation.get_space()).add_constraint(equal)
    return relation.is_subset(projected)


def issue_plan(call: Call, ctx) -> IssuePlan:
    """Derive the repeat nest and operand types for one instruction issue."""
    plan, _single = _issue_plan(call, ctx)
    return plan


def _outer_band(
    relations: AccessRelations,
    repeat: tuple[int, ...],
    order: tuple[int, ...],
    single_shape: tuple[int, ...],
) -> AccessRelations:
    """Compose repeat coordinates with one issue's affine coordinate equations."""
    if all(count == 1 for count in repeat):
        return relations
    outer_axes = tuple(order)
    outer = tuple(f"r{axis}" for axis in outer_axes)
    inner = tuple(f"d{axis}" for axis in range(len(repeat)))
    global_ = tuple(f"g{axis}" for axis in range(len(repeat)))
    domain = (*outer, *inner)
    constraints = [
        *(f"0 <= r{axis} < {repeat[axis]}" for axis in outer_axes),
        *(f"0 <= d{axis} < {single_shape[axis]}" for axis in range(len(repeat))),
        *(f"g{axis} = {single_shape[axis]} * r{axis} + d{axis}" for axis in range(len(repeat))),
    ]
    band = isl.map(
        f"{{ [{', '.join(domain)}] -> [{', '.join(global_)}] : {' and '.join(constraints)} }}"
    )

    def lifted(boundary: BoundaryRelation) -> BoundaryRelation:
        pattern = boundary.pattern
        relation = band.apply_range(relation_of(pattern).affine_hull())
        parameters = dict(pattern.parameters)
        return BoundaryRelation(
            AffineAccess(
                relation,
                tuple(
                    (name, parameters[name])
                    for name in (
                        relation.get_dim_name(isl.dim_type.PARAM, index)
                        for index in range(relation.dim(isl.dim_type.PARAM))
                    )
                ),
            )
        )

    return AccessRelations(
        inputs=tuple(lifted(boundary) for boundary in relations.inputs),
        outputs=tuple(lifted(boundary) for boundary in relations.outputs),
    )


@register_access_relation(ScheduleOp)
def _schedule_access_relation(call: Call, ctx) -> AccessRelations:
    plan, single = _issue_plan(call, ctx)
    params, _reads, _writes = _instruction_schema(call.target.op)
    scheduled = _outer_band(single, plan.repeat, plan.order, plan.single_shape)
    return AccessRelations(
        inputs=tuple(
            boundary
            for param, boundary in zip(params, scheduled.inputs, strict=True)
            if param.effect & MemoryEffect.READ
        ),
        outputs=scheduled.outputs,
    )


@register_cost_evaluator(ScheduleOp)
def _schedule_cost(call: Call, ctx) -> Cost:
    """Count reached bytes and contraction work in the selected local view."""
    stated = relations_of(call, ctx)
    local = projected(stated, call, ctx)
    types = (*(ctx.local_type_of(arg) for arg in call.args), ctx.local_output_type(call))
    boundaries = (*local.inputs, *local.outputs)
    moved = []
    for index, (type_, boundary) in enumerate(zip(types, boundaries, strict=True)):
        if not isinstance(type_, TensorType):
            raise ValueError("ScheduleOp cost requires tensor operands and output")
        elements = reached_elements(boundary.pattern)
        if elements is None:
            raise ValueError(f"ScheduleOp boundary {index} has no finite traffic")
        amount = -(-(elements * type_.dtype.bit_width) // 8)
        moved.append(
            TrafficBytes(write=amount) if index == len(call.args) else TrafficBytes(read=amount)
        )

    op = call.target.op
    _params, reads, writes = _instruction_schema(op)
    read_write = tuple(param for param in writes if param.effect & MemoryEffect.READ)
    flops = {}
    if read_write:
        iterations = cardinality(iteration_universe(local))
        if iterations is None:
            raise ValueError("ScheduleOp has no finite floating-point iteration count")
        dtype = next(
            (
                type_.dtype
                for param, type_ in zip(reads, types, strict=False)
                if not param.effect & MemoryEffect.WRITE and isinstance(type_, TensorType)
            ),
            types[-1].dtype,
        )
        flops = {dtype: 2 * iterations}
    return Cost(flops, tuple(moved))


def _check_pattern(
    call: Call,
    ctx,
    matcher: PatternMatcher,
    param: ParamDef,
    pattern: TensorPattern | None,
    type_: TensorType,
) -> None:
    if pattern is None or matcher.match(pattern, type_):
        return
    from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

    ctx.error(
        call,
        f"{type(call.target.op).__name__} {param.name} single-issue tile does not match: "
        f"{PatternPrinter().refusal(matcher.refusal)}",
    )


@register_typeinfer(ScheduleOp)
def _infer_schedule(call: Call, ctx) -> TensorType:
    schedule = call.target
    op = schedule.op
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        ctx.error(call, f"{type(op).__name__} is not a registered operation")
    params = tuple(param for param in schema.signature if param.kind == "input")
    if any(param.effect is None for param in params):
        ctx.error(call, f"{type(op).__name__} does not declare every operand's memory effect")
    reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
    writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    if len(call.args) != len(reads):
        ctx.error(
            call,
            f"{type(op).__name__} reads {len(reads)} operands, got {len(call.args)}",
        )
    if not writes:
        ctx.error(call, f"{type(op).__name__} writes no result operand")

    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    patterns = {param.name: _operand_pattern(param, op, bindings) for param in params}
    whole: dict[str, TensorType] = {
        param.name: ctx.type_of(arg) for param, arg in zip(reads, call.args, strict=True)
    }
    arranged = {
        param.name: _as_it_lies(arg, whole[param.name], ctx)
        for param, arg in zip(reads, call.args, strict=True)
    }
    results: dict[str, TensorType] = {}
    for param in writes:
        if param.effect & MemoryEffect.READ:
            results[param.name] = whole[param.name]
        else:
            pattern = patterns[param.name]
            if pattern is None:
                ctx.error(call, f"{type(op).__name__} {param.name} has no tensor pattern")
            try:
                results[param.name] = _result_type(
                    op, param, pattern, whole, bindings, ctx.current_mesh
                )
            except (TypeError, ValueError) as error:
                ctx.error(call, str(error))
        whole[param.name] = results[param.name]

    match_types = {**whole, **arranged}
    try:
        plan, _single_relations = _derive_issue(
            schedule,
            op,
            tuple(patterns[param.name] for param in params),
            tuple(match_types[param.name] for param in params),
            bindings,
            ctx.current_mesh,
        )
    except (TypeError, ValueError) as error:
        ctx.error(call, str(error))
    single = dict(zip((param.name for param in params), plan.operand_types, strict=True))
    matcher = PatternMatcher(bindings)
    for param in params:
        _check_pattern(call, ctx, matcher, param, patterns[param.name], single[param.name])
    if not matcher.solve():
        from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

        ctx.error(call, PatternPrinter().refusal(matcher.refusal))
    for rule in between_rules(type(op)):
        if not rule.holds(single):
            ctx.error(call, rule.refused(single))

    return results[writes[0].name]


__all__ = ["IssueAxis", "IssuePlan", "ScheduleOp", "issue_plan"]
