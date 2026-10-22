"""One admission check and tile projection for HIR and TIR calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import prod

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    Op,
    Var,
    VerifyError,
    capabilities_of,
    op_identifier,
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
from tilefoundry.ir.types import ComposedLayout, Layout, Split, TensorType, UnitType
from tilefoundry.ir.types.int_tuple import flatten, partition_modes
from tilefoundry.ir.types.layout import get
from tilefoundry.ir.types.layout_algebra import composition, zipped_divide
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
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
        self.instruction = "" if op is None else op_identifier(declaration)
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
    operands: tuple
    repeat: tuple
    order: tuple
    shape: tuple
    issued: tuple = ()
    bind: tuple = ()
    repeat_layouts: tuple = ()


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


@dataclass
class _AxisTile:
    """One row-major axis split, with original mode positions for ownership."""

    inner: Layout
    repeat: Layout
    logical: Layout
    prefix: tuple
    suffix: tuple
    owners: tuple
    bound: tuple = ()

    def align(self, shape, bound, name):
        if tuple(self.repeat.shape) != shape:
            if any(self.owners):
                raise ValueError(
                    f"{name}: bound repeat modes {self.repeat.shape} do not match {shape}"
                )
            profile = Layout(shape, compact_row_major(shape))
            try:
                self.repeat = composition(self.repeat, profile, major="row")
                self.logical = composition(self.logical, profile, major="row")
            except (AssertionError, ValueError) as error:
                raise ValueError(
                    f"{name}: repeat mode {self.repeat.shape} cannot factor as {shape}"
                ) from error
            if self.repeat.shape != shape or self.logical.shape != shape:
                raise ValueError(
                    f"{name}: repeat mode {self.repeat.shape} cannot factor as {shape}"
                )
            self.prefix = (None,) * len(shape)
            self.owners = (False,) * len(shape)
        if any(owner and not shared for owner, shared in zip(self.owners, bound, strict=True)):
            raise ValueError(f"{name}: repeat mode has a conflicting Split")
        self.bound = bound


def _axis_tile(layout, logical_extent, tile, positions, split, name):
    """Divide using the authored row-major inner profile for TensorType readers."""
    extents, strides = tuple(flatten(layout.shape)), tuple(flatten(layout.strides))
    if logical_extent == tile:
        unit = Layout((1,), (0,))
        return _AxisTile(layout, unit, unit, (None,), tuple(positions), (False,))
    count = logical_extent // tile
    try:
        leading, trailing = partition_modes(zip(positions, extents, strict=True), count, split)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    prefix, prefix_widths = zip(*leading)
    suffix, suffix_widths = zip(*(trailing or ((None, 1),)))
    tiler = Layout(tuple(suffix_widths), compact_row_major(tuple(suffix_widths)))
    divided = zipped_divide(layout, tiler, major="row")
    inner = get(divided, 0)
    if inner.shape != tuple(suffix_widths):
        raise ValueError(
            f"{name}: inner mode {inner.shape} does not preserve {tuple(suffix_widths)}"
        )
    original = dict(zip(positions, strides, strict=True))
    inner = Layout(
        inner.shape,
        tuple(
            original[position] if width == 1 and position is not None else stride
            for position, width, stride in zip(suffix, inner.shape, inner.strides, strict=True)
        ),
    )
    repeat_profile = Layout(tuple(prefix_widths), compact_row_major(tuple(prefix_widths)))
    repeat = composition(get(divided, 1), repeat_profile, major="row")
    logical = composition(
        get(zipped_divide(Layout((logical_extent,), (1,)), tiler, major="row"), 1),
        repeat_profile,
        major="row",
    )
    if repeat.shape != tuple(prefix_widths) or logical.shape != tuple(prefix_widths):
        raise ValueError(f"{name}: repeat mode cannot preserve prefix {tuple(prefix_widths)}")
    return _AxisTile(
        inner,
        repeat,
        logical,
        tuple(prefix),
        tuple(suffix),
        tuple(position in split for position in prefix),
    )


class _OperandTile:
    def __init__(self, type_, shape, participant, enclosing, attrs, name):
        self.type, self.shape = type_, shape
        self.participant, self.enclosing, self.attrs = participant, enclosing, attrs
        self.top = type_.layout if isinstance(type_.layout, ComposedLayout) else None
        outer = self.top.outer if self.top is not None else type_.layout
        self.shard = outer if isinstance(outer, ShardLayout) else None
        self.address = self.shard.layout if self.shard is not None else outer
        primitive = self.address.outer if isinstance(self.address, ComposedLayout) else self.address
        if not isinstance(primitive, Layout):
            raise ValueError(f"{name}: cannot divide {type(primitive).__name__}")
        self.grouped = any(isinstance(mode, tuple) for mode in primitive.shape)
        correspondence = layout_correspondence(primitive.shape, type_.shape)
        if correspondence.groups is None:
            raise ValueError(f"{name}: modes cross logical axes {type_.shape}")
        split = (
            {attr.axis for attr in self.shard.attrs if isinstance(attr, Split)}
            if self.shard
            else set()
        )
        self.axes = []
        for axis, (group, extent, tile) in enumerate(
            zip(correspondence.groups, type_.shape, shape, strict=True)
        ):
            mode = correspondence.axis_mode(primitive.shape, axis)
            stride = correspondence.axis_mode(primitive.strides, axis)
            layout = Layout(
                mode if isinstance(mode, tuple) else (mode,),
                stride if isinstance(stride, tuple) else (stride,),
            )
            self.axes.append(_axis_tile(layout, extent, tile, group, split, f"{name} axis {axis}"))

    def build(self, *, issued=False):
        shapes, strides, mapping, shape = [], [], {}, []
        for axis, tile in enumerate(self.axes):
            prefix = tuple(
                index
                for index, (width, bound) in enumerate(
                    zip(tile.repeat.shape, tile.bound, strict=True)
                )
                if issued and (bound or (width == 1 and tile.prefix[index] is not None))
            )
            prefix_shape = tuple(tile.repeat.shape[index] for index in prefix)
            part = tile.inner
            if prefix:
                part = Layout(
                    prefix_shape + tuple(flatten(part.shape)),
                    tuple(tile.repeat.strides[index] for index in prefix)
                    + tuple(flatten(part.strides)),
                )
            origin = sum(len(tuple(flatten(mode))) for mode in shapes)
            for index, selected in enumerate(prefix):
                if tile.prefix[selected] is not None:
                    mapping[tile.prefix[selected]] = origin + index
            for index, position in enumerate(tile.suffix):
                if position is not None:
                    mapping[position] = origin + len(prefix) + index
            if self.grouped:
                shapes.append(part.shape if len(part.shape) > 1 else part.shape[0])
                strides.append(part.strides if len(part.strides) > 1 else part.strides[0])
            else:
                shapes.extend(flatten(part.shape))
                strides.extend(flatten(part.strides))
            width = prod(prefix_shape)
            shape.append(self.shape[axis] if width == 1 else self.shape[axis] * width)
        layout = Layout(tuple(shapes), tuple(strides))
        broadcast = self.shard is None or all(
            isinstance(attr, Broadcast) for attr in self.shard.attrs
        )
        if isinstance(self.address, ComposedLayout):
            layout = replace(
                self.address,
                outer=layout,
                offset=0 if not issued and broadcast else self.address.offset,
            )
        if self.top is not None and self.top.inner is not None:
            layout = ComposedLayout(self.top.inner, 0, layout)
        if self.shard is not None and (issued or not broadcast):
            source = self.shard if issued else participant_layout(self.shard, self.participant)
            frame, dropped = (
                (source.mesh, 0) if issued else _participant_frame(source.mesh, self.participant)
            )
            attrs = tuple(
                Split(mapping[attr.axis]) if isinstance(attr, Split) else attr
                for attr in source.attrs[dropped:]
            )
            layout = ShardLayout(layout, attrs, frame)
        elif self.attrs is not None:
            frame = separate(self.enclosing)[-1]
            participant, dropped = _participant_frame(frame, self.participant)
            layout = ShardLayout(
                layout,
                (Broadcast(),) * dropped + self.attrs if issued else self.attrs,
                frame if issued else participant,
            )
        return TensorType(tuple(shape), self.type.dtype, layout, self.type.storage)


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
        return Projection(params, writes, patterns, tuple(operand_types), (), (), ())
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
    if getattr(op, "atom", None) is None:
        return Projection(params, writes, patterns, whole_types, repeat, order, single_shape)
    atom = getattr(op, "atom", None)
    participant = (
        atom.execution_mesh
        if isinstance(atom, type)
        else getattr(atom, "required_execution_mesh", None)
    )
    if not fragments:
        inner_types = tuple(
            TensorType(shape, type_.dtype, type_.layout, type_.storage)
            for shape, type_ in zip(shapes, whole_types, strict=True)
        )
        return Projection(params, writes, patterns, inner_types, repeat, order, single_shape)
    tiles, axes = [], []
    try:
        for param, type_, shape, pattern, boundary in zip(
            params, whole_types, shapes, patterns, single[: len(params)], strict=True
        ):
            attrs = (
                pattern.layout.attrs
                if pattern is not None and isinstance(pattern.layout, ShardLayoutPattern)
                else None
            )
            tiles.append(_OperandTile(type_, shape, participant, mesh, attrs, param.name))
            axes.append(projected_axes(boundary))
        shared = {}
        for param, tile, mapped in zip(params, tiles, axes, strict=True):
            for axis, one in zip(mapped, tile.axes, strict=True):
                if axis is None or not any(one.owners):
                    continue
                value = (one.repeat.shape, one.owners)
                if axis in shared and shared[axis] != value:
                    raise ValueError(
                        f"{param.name}: repeat mode on iteration axis {axis} has conflicting Split extents"
                    )
                shared[axis] = value
        for param, tile, mapped in zip(params, tiles, axes, strict=True):
            for axis, one in zip(mapped, tile.axes, strict=True):
                if axis in shared:
                    one.align(*shared[axis], f"{param.name} iteration axis {axis}")
                else:
                    one.bound = (False,) * len(one.repeat.shape)
        inner_types = tuple(tile.build() for tile in tiles)
        issued = tuple(tile.build(issued=True) for tile in tiles)
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=bindings, op=op) from error
    bind = tuple(
        prod(width for width, bound in zip(*shared[axis], strict=True) if bound)
        if axis in shared
        else 1
        for axis in range(len(repeat))
    )
    repeat_layouts = tuple(
        tuple((axis, one.logical, one.bound) for axis, one in zip(mapped, tile.axes, strict=True))
        for mapped, tile in zip(axes, tiles, strict=True)
    )
    return Projection(
        params,
        writes,
        patterns,
        inner_types,
        repeat,
        order,
        single_shape,
        issued,
        bind,
        repeat_layouts,
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
    for param, pattern, type_ in zip(params, tiles.patterns, tiles.operands, strict=True):
        match_type = type_ if view is None else view(param.name, type_)
        if pattern is not None and not matcher.match(pattern, match_type):
            raise _pattern_refusal(matcher, op, param.name)
    named = dict(zip((p.name for p in params), tiles.operands, strict=True))
    for rule in between_rules(type(op)):
        if not rule.holds(named):
            raise ScheduleRefusal(rule.refused(named), path=matcher.bindings, op=op)
    if not matcher.solve():
        raise _pattern_refusal(matcher, op)
    if handler is not None:
        args = tuple(
            value if isinstance(value, Constant) else Var(param.name, type=type_)
            for value, param, type_ in zip(operands, params, tiles.operands, strict=True)
        )
        call = Call(op, args, type=UnitType())
        ctx = (
            replace(
                frame,
                memo={
                    id(arg): (arg, type_) for arg, type_ in zip(args, tiles.operands, strict=True)
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
