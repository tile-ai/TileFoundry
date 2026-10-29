"""One admission check and tile projection for HIR and TIR calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import accumulate
from math import prod

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Op,
    Var,
    VerifyError,
    capabilities_of,
    get_registered_name,
    target_capabilities,
)
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.mesh_scope import covered_by_scope, storage_reaches
from tilefoundry.ir.pattern import PatternMatcher, ShardLayoutPattern, TensorPattern, between_rules
from tilefoundry.ir.pattern.utils import (
    declared_execution_mesh,
    declared_shape,
    selected_pattern,
)
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, Split, TensorType, UnitType
from tilefoundry.ir.types.int_tuple import axis_tiler, compatible_profile, flatten, mode_axes
from tilefoundry.ir.types.int_tuple import unflatten as unflatten_tuple
from tilefoundry.ir.types.layout import flatten as flatten_layout
from tilefoundry.ir.types.layout import get, unflatten
from tilefoundry.ir.types.layout_algebra import zipped_divide
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    ShardLayout,
    shard_layout_of,
)
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import (
    _participant_frame,
    participant_layout,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    access_relation_registry,
    iteration_universe,
    projected_axes,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import VerifyContext
from tilefoundry.visitor_registry.registries import verify_stmt_registry


class ScheduleRefusal(ValueError):
    """An instruction's failed node and the bindings reached before that refusal."""

    def __init__(self, reason, *, path=(), op=None):
        self.reason = reason
        self.path = tuple(path.items()) if isinstance(path, dict) else tuple(path)
        atom = getattr(op, "atom", None)
        declaration = (
            atom if isinstance(atom, type) else type(atom) if atom is not None else type(op)
        )
        self.instruction = "" if op is None else get_registered_name(declaration)
        super().__init__(reason)

    def __str__(self):
        from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

        printer = PatternPrinter()
        return printer.match_text(
            self.instruction, printer.match_data((self.path,), self.reason)
        ).lstrip()


class _RelationContext:
    def __init__(self, types: dict[int, TensorType]) -> None:
        self._types = types

    def type_of(self, expr: Var) -> TensorType:
        return self._types[id(expr)]


def operand_relations(op: Op, operand_types: tuple[TensorType, ...]) -> tuple[AccessRelation, ...]:
    """The registered access relations of *op* over operands of these types."""
    args = tuple(
        Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(operand_types)
    )
    call = Call(target=op, args=args, type=UnitType())
    return relations_of(call, _RelationContext(dict(zip(map(id, args), operand_types))))


def _operand_pattern(param: ParamDef, op: Op, bindings: dict) -> TensorPattern | None:
    pattern = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
    selected = selected_pattern(pattern, bindings)
    return selected if isinstance(selected, TensorPattern) else None


def _iteration_shape(relations: tuple[AccessRelation, ...]) -> tuple[int, ...]:
    space = iteration_universe(relations)
    if space is None:
        raise ValueError("instruction access relations state no iteration space")
    return tuple(
        int(space.dim_max(axis).max_val().num_si()) + 1 for axis in range(space.tuple_dim())
    )


@dataclass(frozen=True)
class Projection:
    params: tuple
    writes: tuple
    patterns: tuple
    tiles: tuple
    rests: tuple
    repeat: tuple
    order: tuple
    shape: tuple
    logical_rests: tuple = ()
    bind_masks: tuple = ()
    axes: tuple = ()
    serial: tuple = ()


def call_params(op, count):
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        raise ScheduleRefusal(f"{type(op).__name__} is not a registered operation", op=op)
    params = tuple(p for p in schema.signature if p.kind == "input")
    required = sum(not p.optional for p in params)
    if not required <= count <= len(params):
        raise ScheduleRefusal(
            f"{type(op).__name__} expects operands {tuple(p.name for p in params)} ({required}..{len(params)}), got {count}",
            op=op,
        )
    optional = tuple(p for p in params if p.optional)
    included = {id(p) for p in optional[: count - required]}
    return tuple(p for p in params if not p.optional or id(p) in included)


def check_input_reachability(op, operand_types, mesh):
    if mesh is None:
        return
    names = getattr(op, "scope_inputs", None)
    params = getattr(getattr(type(op), "_op_schema", None), "signature", ())
    inputs = tuple(p.name for p in params if p.kind == "input")
    for index, type_ in enumerate(operand_types):
        if names is not None and (index >= len(inputs) or inputs[index] not in names):
            continue
        layout = shard_layout_of(getattr(type_, "layout", None))
        if layout is None:
            continue
        projected = participant_layout(layout)
        if not covered_by_scope(projected.mesh, mesh):
            raise ValueError(
                f"input {index} is laid out more finely than the scope it runs in; write it inside that scope, or reshard it back first"
            )
        if not storage_reaches(type_.storage, projected.mesh, mesh):
            raise ValueError(
                f"input {index} is laid out more coarsely and kept in {type_.storage.name.lower()}, which does not reach the units this runs on; reshard it to smem or gmem first"
            )


def project(op, operand_types, mesh, *, bindings=None, repeat=None, order=None, fragments=True):
    operand_types = tuple(operand_types)
    params = call_params(op, len(operand_types))
    writes = tuple(param for param in params if param.effect and param.effect & MemoryEffect.WRITE)
    bindings = (
        dict(getattr(getattr(op, "atom", None), "bindings", {}))
        if bindings is None
        else dict(bindings)
    )
    patterns = tuple(_operand_pattern(param, op, bindings) for param in params)
    whole = {param.name: type_ for param, type_ in zip(params, operand_types, strict=True)}
    if (
        getattr(op, "atom", None) is None
        and fragments
        and repeat is None
        and order is None
        and not any(declared_shape(pattern, bindings) is not None for pattern in patterns)
    ):
        return Projection(params, writes, patterns, tuple(operand_types), (), (), (), ())
    try:
        whole_relations = operand_relations(op, tuple(whole[param.name] for param in params))
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=bindings, op=op) from error
    whole_shape = _iteration_shape(whole_relations)
    whole_types = tuple(whole[param.name] for param in params)
    shapes = tuple(
        declared_shape(pattern, bindings) or tuple(type_.shape)
        for pattern, type_ in zip(patterns, whole_types, strict=True)
    )
    provisional = tuple(
        TensorType(shape, type_.dtype, type_.layout, type_.storage)
        for shape, type_ in zip(shapes, whole_types, strict=True)
    )
    try:
        single = operand_relations(op, provisional)
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=bindings, op=op) from error
    single_shape = _iteration_shape(single)
    if len(whole_shape) != len(single_shape):
        raise ScheduleRefusal(
            "single-issue and scheduled iteration ranks differ", path=bindings, op=op
        )
    inferred = []
    for whole_extent, single_extent in zip(whole_shape, single_shape, strict=True):
        if single_extent < 1 or whole_extent % single_extent:
            raise ScheduleRefusal(
                f"iteration extent {whole_extent} is not divisible by "
                f"single-issue extent {single_extent}",
                path=bindings,
                op=op,
            )
        inferred.append(whole_extent // single_extent)
    inferred = tuple(inferred)
    repeat = inferred if repeat is None else repeat
    fixed = any(declared_shape(pattern, bindings) is not None for pattern in patterns)
    if not fixed and any(count != 1 for count in repeat):
        raise ScheduleRefusal("transfer tiling is not yet supported", path=bindings, op=op)
    if repeat != inferred:
        raise ScheduleRefusal(
            f"repeat {repeat} conflicts with inferred repeat {inferred}", path=bindings, op=op
        )
    order = tuple(range(len(repeat))) if order is None else order
    if sorted(order) != list(range(len(repeat))):
        raise ScheduleRefusal(
            f"order {order} is not a permutation of iteration dimensions", path=bindings, op=op
        )
    declaration = getattr(op, "atom", op)
    participant = (
        declaration.execution_mesh
        if isinstance(declaration, type)
        else getattr(declaration, "required_execution_mesh", getattr(declaration, "execution_mesh", None))
    )
    if not isinstance(participant, Mesh):
        return Projection(params, writes, patterns, whole_types, (), repeat, order, single_shape)
    if not fragments:
        inner_types = tuple(
            TensorType(shape, type_.dtype, type_.layout, type_.storage)
            for shape, type_ in zip(shapes, whole_types, strict=True)
        )
        return Projection(params, writes, patterns, inner_types, (), repeat, order, single_shape)
    tiles, rests, axes, logical_rests = [], [], [], []
    try:
        for param, type_, shape, pattern, boundary in zip(
            params, whole_types, shapes, patterns, single[: len(params)], strict=True
        ):
            attrs = (
                pattern.layout.attrs
                if pattern is not None and isinstance(pattern.layout, ShardLayoutPattern)
                else None
            )
            name = param.name
            top = type_.layout if isinstance(type_.layout, ComposedLayout) else None
            outer = top.outer if top is not None else type_.layout
            shard = outer if isinstance(outer, ShardLayout) else None
            address = shard.layout if shard is not None else outer
            primitive = address.outer if isinstance(address, ComposedLayout) else address
            if not isinstance(primitive, Layout):
                raise ValueError(f"{name}: cannot divide {type(primitive).__name__}")
            profile = compatible_profile(primitive.shape, type_.shape)
            grouped = unflatten(flatten_layout(primitive), profile)
            split = (
                {attr.axis for attr in shard.attrs if isinstance(attr, Split)} if shard else set()
            )
            try:
                tiler = axis_tiler(profile, shape, splits=split)
            except ValueError as error:
                first = 0
                for axis, (mode, size) in enumerate(zip(profile, shape, strict=True)):
                    widths = flatten(mode)
                    count = prod(widths) // size
                    for index, prefix in enumerate(accumulate(widths, lambda x, y: x * y)):
                        if count % prefix and prefix % count:
                            raise ValueError(
                                f"{name} axis {axis}: repeat is not leftmost at mode {first + index} "
                                f"(extent {widths[index]})"
                            ) from error
                    first += len(widths)
                raise ValueError(f"{name}: {error}") from error
            compact = Layout(
                profile,
                tuple(
                    compact_row_major(mode) if isinstance(mode, tuple) else 1 for mode in profile
                ),
            )
            logical_rest = get(zipped_divide(compact, tiler, major="row"), 1)
            divided = zipped_divide(grouped, tiler, major="row")
            tile, rest = get(divided, 0), get(divided, 1)
            flat, remainder = flatten_layout(tile), flatten_layout(rest)
            flat = replace(
                flat,
                strides=tuple(
                    original if extent == 1 else stride
                    for extent, original, stride in zip(
                        flatten(primitive.shape),
                        flatten(primitive.strides),
                        flat.strides,
                        strict=True,
                    )
                ),
            )
            layout_axes = mode_axes(profile)
            kept = tuple(
                i
                for i, (before, after) in enumerate(
                    zip(flatten(primitive.shape), flat.shape, strict=True)
                )
                if before == 1 or after != 1 or layout_axes.count(layout_axes[i]) == 1
            )
            mapping = {old: new for new, old in enumerate(kept)}
            if any(isinstance(mode, tuple) for mode in primitive.shape):
                parts = tuple(
                    (
                        tuple(flat.shape[i] for i in kept if layout_axes[i] == axis),
                        tuple(flat.strides[i] for i in kept if layout_axes[i] == axis),
                    )
                    for axis in range(len(shape))
                )
                tile = Layout(
                    tuple(s[0] if len(s) == 1 else s for s, _d in parts),
                    tuple(d[0] if len(d) == 1 else d for _s, d in parts),
                )
            else:
                tile = Layout(
                    tuple(flat.shape[i] for i in kept), tuple(flat.strides[i] for i in kept)
                )
            broadcast = shard is None or all(isinstance(attr, Broadcast) for attr in shard.attrs)
            if isinstance(address, ComposedLayout):
                tile = replace(address, outer=tile, offset=0 if broadcast else address.offset)
            if top is not None and top.inner is not None:
                tile = ComposedLayout(top.inner, 0, tile)
            if shard is not None:
                tile_attrs = tuple(
                    Split(mapping[attr.axis])
                    if isinstance(attr, Split)
                    and attr.axis in mapping
                    and remainder.shape[attr.axis] == 1
                    else Broadcast()
                    if isinstance(attr, Split)
                    else attr
                    for attr in shard.attrs
                )
                source = participant_layout(replace(shard, attrs=tile_attrs), participant)
                frame, dropped = _participant_frame(source.mesh, participant)
                tile = ShardLayout(tile, source.attrs[dropped:], frame)
                rest = ShardLayout(
                    rest,
                    tuple(
                        Split(attr.axis)
                        if isinstance(attr, Split) and remainder.shape[attr.axis] != 1
                        else Broadcast()
                        for attr in shard.attrs
                    ),
                    shard.mesh,
                )
            elif attrs is not None:
                frame = separate(participant if mesh is None else mesh)[-1]
                selected, _dropped = _participant_frame(frame, participant)
                tile = ShardLayout(tile, attrs, selected)
            tile = TensorType(shape, type_.dtype, tile, type_.storage)
            mapped = projected_axes(boundary)
            physical = rest.layout if isinstance(rest, ShardLayout) else rest
            profile = compatible_profile(
                physical.shape, tuple(1 if axis is None else repeat[axis] for axis in mapped)
            )
            regrouped = unflatten(flatten_layout(physical), profile)
            rests.append(
                replace(rest, layout=regrouped) if isinstance(rest, ShardLayout) else regrouped
            )
            logical_rests.append(unflatten(flatten_layout(logical_rest), regrouped.shape))
            tiles.append(tile)
            axes.append(mapped)
        shared = {}
        for param, rest, mapped in zip(params, rests, axes, strict=True):
            if not isinstance(rest, ShardLayout):
                continue
            flat = flatten_layout(rest.layout)
            logical_axes = mode_axes(rest.shape)
            for axis in set(logical_axes):
                iteration_axis = mapped[axis]
                if iteration_axis is None:
                    continue
                modes = tuple(
                    i for i, a in enumerate(logical_axes) if a == axis and flat.shape[i] != 1
                )
                splits = tuple(
                    (modes.index(attr.axis), flat.shape[attr.axis])
                    for attr in rest.attrs
                    if isinstance(attr, Split) and attr.axis in modes
                )
                if not splits:
                    continue
                if iteration_axis in shared and shared[iteration_axis][0] != splits:
                    raise ValueError(
                        f"{param.name}: repeat mode on iteration axis {iteration_axis} has conflicting Split extents"
                    )
                shared[iteration_axis] = (
                    splits,
                    tuple(flat.shape[i] for i in modes),
                    rest.mesh,
                    tuple(
                        (mesh_axis, modes.index(attr.axis))
                        for mesh_axis, attr in enumerate(rest.attrs)
                        if isinstance(attr, Split) and attr.axis in modes
                    ),
                )
        for param, rest, mapped in zip(params, rests, axes, strict=True):
            physical = rest.layout if isinstance(rest, ShardLayout) else rest
            for axis, iteration_axis in enumerate(mapped):
                if iteration_axis not in shared:
                    continue
                mode = get(physical, axis)
                active = tuple(width for width in flatten(mode.shape) if width != 1)
                expected = shared[iteration_axis][1]
                try:
                    compatible_profile(active, expected)
                except ValueError as error:
                    raise ValueError(
                        f"{param.name} iteration axis {iteration_axis}: repeat mode {active} cannot factor as {expected}"
                    ) from error
        for index, (rest, mapped) in enumerate(zip(rests, axes, strict=True)):
            physical = rest.layout if isinstance(rest, ShardLayout) else rest
            bindings_on_axes = [
                (axis, shared[iteration_axis])
                for axis, iteration_axis in enumerate(mapped)
                if iteration_axis in shared
            ]
            if not bindings_on_axes:
                continue
            rest_mesh = rest.mesh if isinstance(rest, ShardLayout) else bindings_on_axes[0][1][2]
            rest_attrs = (
                list(rest.attrs)
                if isinstance(rest, ShardLayout)
                else [Broadcast()] * len(flatten(rest_mesh.layout.shape))
            )
            logical_axes = mode_axes(physical.shape)
            flat = flatten_layout(physical)
            for axis, (_splits, _shape, source_mesh, bound) in bindings_on_axes:
                modes = [i for i, a in enumerate(logical_axes) if a == axis and flat.shape[i] != 1]
                shift = len(rest_attrs) - len(flatten(source_mesh.layout.shape))
                for mesh_axis, mode in bound:
                    rest_attrs[mesh_axis + shift] = Split(modes[mode])
            rests[index] = ShardLayout(physical, tuple(rest_attrs), rest_mesh)
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=bindings, op=op) from error
    bind_masks = []
    for rest in rests:
        physical = rest.layout if isinstance(rest, ShardLayout) else rest
        bound = (
            {attr.axis for attr in rest.attrs if isinstance(attr, Split)}
            if isinstance(rest, ShardLayout)
            else set()
        )
        bind_masks.append(
            unflatten_tuple(
                tuple(index in bound for index in range(len(flatten(physical.shape)))),
                physical.shape,
            )
        )
    serial = tuple(
        count // prod(width for _position, width in shared[axis][0]) if axis in shared else count
        for axis, count in enumerate(repeat)
    )
    return Projection(
        params,
        writes,
        patterns,
        tuple(tiles),
        tuple(rests),
        repeat,
        order,
        single_shape,
        tuple(logical_rests),
        tuple(bind_masks),
        tuple(axes),
        serial,
    )


def _pattern_refusal(matcher: PatternMatcher, op, parameter=None) -> ScheduleRefusal:
    from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

    refusal = matcher.refusal
    path = tuple((refusal.bindings if refusal is not None else matcher.bindings).items())
    return ScheduleRefusal(PatternPrinter().refusal_reason(refusal, parameter), path=path, op=op)


def _selects(op, capability) -> bool:
    """Whether *op*'s attribute, as a class or an instance, names *capability*'s declaration."""
    value = getattr(op, capability.attribute)
    declaration = value if isinstance(value, type) else type(value)
    return issubclass(declaration, capability.declaration)


def _admit_capabilities(op, target) -> None:
    """Refuse *op* if the target's architecture does not list a capability the op uses."""
    if target is None:
        return
    listed = target_capabilities(target)
    for capability in capabilities_of(type(op)):
        if capability.name is None or capability.name in listed:
            continue
        if capability.attribute is not None and not _selects(op, capability):
            continue
        raise ScheduleRefusal(
            f"needs {capability.name}, which {target.identity} does not list", op=op
        )


def admit_fields(op, operands, mesh, matcher, *, target=None):
    """Admit attributes, execution mesh and operand fields before tile projection."""
    matcher.bindings.update(getattr(op, "bindings", {}))
    _admit_capabilities(op, target)
    for param in type(op)._op_schema.signature:
        if param.kind != "attribute" or param.name == "execution_mesh":
            continue
        value = getattr(op, param.name)
        if param.name == "atom" and isinstance(value, type) and value in param.pattern.declarations:
            continue
        if value is None and param.optional:
            continue
        if (
            isinstance(param.annotation, type)
            and issubclass(param.annotation, Enum)
            and not isinstance(value, param.annotation)
        ):
            raise ScheduleRefusal(
                f"{param.name} must be {param.annotation.__name__}, got {type(value).__name__}",
                op=op,
            )
        if param.annotation is int and not isinstance(value, int):
            raise ScheduleRefusal(f"{param.name} must be int, got {type(value).__name__}", op=op)
        if param.pattern is not None and not matcher.match(param.pattern, value):
            raise _pattern_refusal(matcher, op, param.name)
    if not matcher.match(declared_execution_mesh(op, target), mesh):
        raise _pattern_refusal(matcher, op, "mesh")
    for param, type_ in zip(call_params(op, len(operands)), operands, strict=True):
        pattern = _operand_pattern(param, op, matcher.bindings)
        if isinstance(pattern, TensorPattern):
            pattern = replace(pattern, shape=None, layout=None)
        if pattern is not None and not matcher.match(pattern, type_):
            raise _pattern_refusal(matcher, op, param.name)


def check_call(op, operands, mesh, *, frame, repeat=None, order=None, projection=None):
    """Match whole fields, project atom tiles, then check one closed declaration.

    Operands may be types for candidate admission or typed expressions for
    verification; expressions retain literal values needed by leaf verifiers.
    """
    params = call_params(op, len(operands))
    if access_relation_registry.lookup(type(op)) is None:
        raise ScheduleRefusal(
            f"{type(op).__name__} states no access relations; register one with register_access_relation",
            op=op,
        )
    handler = verify_stmt_registry.lookup(type(op))
    if (
        handler is None
        and not any(param.pattern is not None for param in params)
        and not between_rules(type(op))
    ):
        raise ScheduleRefusal(
            f"{type(op).__name__} has no input pattern, between relation, or registered verifier",
            op=op,
        )
    whole = tuple(value.type if isinstance(value, Expr) else value for value in operands)
    atom = getattr(op, "atom", None)
    matcher = PatternMatcher(dict(getattr(atom, "bindings", {})))
    admit_fields(op, whole, mesh, matcher, target=None if frame is None else frame.target)
    try:
        tiles = (
            projection
            if projection is not None
            else project(op, whole, mesh, bindings=matcher.bindings, repeat=repeat, order=order)
        )
    except ScheduleRefusal:
        raise
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    try:
        check_input_reachability(op, whole, mesh)
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    view = getattr(atom, "match_view", None)
    for param, pattern, type_ in zip(params, tiles.patterns, tiles.tiles, strict=True):
        match_type = type_ if view is None else view(param.name, type_)
        if pattern is not None and not matcher.match(pattern, match_type):
            raise _pattern_refusal(matcher, op, param.name)
    named = dict(zip((p.name for p in params), tiles.tiles, strict=True))
    for rule in between_rules(type(op)):
        if not rule.holds(named):
            raise ScheduleRefusal(rule.refused(named), path=matcher.bindings, op=op)
    if matcher.solve() is not True:
        raise _pattern_refusal(matcher, op)
    if handler is not None:
        args = tuple(
            value if isinstance(value, Constant) else Var(param.name, type=type_)
            for value, param, type_ in zip(operands, params, tiles.tiles, strict=True)
        )
        call = Call(op, args, type=UnitType())
        ctx = (
            replace(
                frame,
                memo={
                    id(arg): (arg, type_) for arg, type_ in zip(args, tiles.tiles, strict=True)
                },
                current_mesh=mesh,
            )
            if isinstance(frame, VerifyContext)
            else VerifyContext(
                module=None if frame is None else frame.module, function=None if frame is None else frame.function, target=None if frame is None else frame.target, topologies=None if frame is None else frame.topologies, current_mesh=mesh, mesh_scope=() if mesh is None else (mesh,)
            )
        )
        try:
            handler(call, ctx)
        except VerifyError as error:
            raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    return dict(matcher.bindings), tiles
