"""One admission check and tile projection for HIR and TIR calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tilefoundry.ir.core import Call, Constant, Expr, Op, Var, VerifyError, op_identifier
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.mesh_scope import covered_by_scope, storage_reaches
from tilefoundry.ir.pattern import PatternMatcher, ShardLayoutPattern, TensorPattern, between_rules
from tilefoundry.ir.pattern.utils import (
    declared_execution_mesh,
    declared_shape,
    matched_row_issues,
    selected_pattern,
)
from tilefoundry.ir.types import TensorType, UnitType
from tilefoundry.ir.types.layout import flatten, resolve_layout
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
from tilefoundry.ir.types.mesh import separate
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
    shard_layout_of,
)
from tilefoundry.ir.types.utils import (
    _participant_frame,
    participant_layout,
    tile_inner_type,
    tile_view_layout,
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
    parallel: tuple = ()
    rows: tuple = ()


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
    operand_types = tuple(
        replace(t, layout=resolve_layout(t.layout, t.shape)) if isinstance(t, TensorType) else t
        for t in operand_types
    )
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
    inner_types = []
    for type_, shape, pattern, boundary in zip(
        whole_types, shapes, patterns, single[: len(params)], strict=True
    ):
        if not fragments:
            inner_types.append(TensorType(shape, type_.dtype, type_.layout, type_.storage))
            continue
        counts = tuple(1 if axis is None else repeat[axis] for axis in projected_axes(boundary))
        shard_attrs = (
            pattern.layout.attrs
            if pattern is not None and isinstance(pattern.layout, ShardLayoutPattern)
            else None
        )
        try:
            inner = tile_inner_type(
                type_,
                shape,
                counts=counts,
                participant=participant,
                enclosing=mesh,
                shard_attrs=shard_attrs,
            )
        except ValueError as error:
            raise ScheduleRefusal(str(error), path=bindings, op=op) from error
        inner_types.append(inner)
    inner_types = tuple(inner_types)
    if not fragments:
        return Projection(params, writes, patterns, inner_types, repeat, order, single_shape)
    parallel = [1] * len(repeat)
    if fragments:
        for type_, boundary in zip(whole_types, single[: len(params)], strict=True):
            shard = shard_layout_of(type_.layout)
            if shard is None:
                continue
            shard = participant_layout(shard, participant)
            physical = flatten(shard.mesh.layout)
            _frame, dropped = _participant_frame(shard.mesh, participant)
            corresponding = layout_correspondence(shard.layout.shape, type_.shape)
            tensor_axes = corresponding.tensor_axes()
            axes = projected_axes(boundary)
            shares = [1] * len(repeat)
            for mesh_axis in range(dropped):
                count = physical.shape[mesh_axis]
                attr = shard.attrs[mesh_axis]
                if count == 1 or not isinstance(attr, Split):
                    continue
                axis = axes[tensor_axes[attr.axis]]
                if axis is None or repeat[axis] % count:
                    raise ScheduleRefusal(
                        f"outer mesh mode {mesh_axis} of extent {count} does not divide its logical repeat",
                        path=bindings,
                        op=op,
                    )
                shares[axis] *= count
            for axis, count in enumerate(shares):
                if repeat[axis] % count:
                    raise ScheduleRefusal(
                        f"outer mesh extent {count} does not divide logical repeat {repeat[axis]}",
                        path=bindings,
                        op=op,
                    )
                parallel[axis] = max(parallel[axis], count)
    issued = []
    for type_, pattern, boundary in zip(whole_types, patterns, single[: len(params)], strict=True):
        axes = projected_axes(boundary)
        counts = tuple(1 if axis is None else repeat[axis] // parallel[axis] for axis in axes)
        shape = tuple(extent // count for extent, count in zip(type_.shape, counts, strict=True))
        attrs = (
            pattern.layout.attrs
            if pattern is not None and isinstance(pattern.layout, ShardLayoutPattern)
            else None
        )
        layout = tile_view_layout(
            type_,
            shape,
            counts=counts,
            enclosing=mesh,
            shard_attrs=attrs,
            participant=None if shard_layout_of(type_.layout) is not None else participant,
        )
        if isinstance(layout, ShardLayout) and shard_layout_of(type_.layout) is None:
            frame = separate(mesh)[-1]
            _participant, dropped = _participant_frame(frame, participant)
            layout = replace(layout, attrs=(Broadcast(),) * dropped + layout.attrs, mesh=frame)
        issued.append(TensorType(shape, type_.dtype, layout, type_.storage))
    return Projection(
        params,
        writes,
        patterns,
        inner_types,
        repeat,
        order,
        single_shape,
        tuple(issued),
        tuple(parallel),
    )


def _pattern_refusal(matcher: PatternMatcher, op, parameter=None) -> ScheduleRefusal:
    from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

    refusal = matcher.refusal
    path = tuple((refusal.bindings if refusal is not None else matcher.bindings).items())
    return ScheduleRefusal(PatternPrinter().refusal_reason(refusal, parameter), path=path, op=op)


def admit_fields(op, operands, mesh, matcher):
    """Admit attributes, execution mesh and operand fields before tile projection."""
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
    if not matcher.match(declared_execution_mesh(op), mesh):
        raise _pattern_refusal(matcher, op, "mesh")
    for param, type_ in zip(call_params(op, len(operands)), operands, strict=True):
        pattern = _operand_pattern(param, op, matcher.bindings)
        if isinstance(pattern, TensorPattern):
            pattern = replace(pattern, shape=None, layout=None)
        if pattern is not None and not matcher.match(pattern, type_):
            raise _pattern_refusal(matcher, op, param.name)


def check_call(op, operands, mesh, *, frame, repeat=None, order=None):
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
    whole = tuple(
        replace(t, layout=resolve_layout(t.layout, t.shape)) if isinstance(t, TensorType) else t
        for t in whole
    )
    atom = getattr(op, "atom", None)
    matcher = PatternMatcher(dict(getattr(atom, "bindings", {})))
    admit_fields(op, whole, mesh, matcher)
    try:
        tiles = project(op, whole, mesh, bindings=matcher.bindings, repeat=repeat, order=order)
    except ScheduleRefusal:
        raise
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    try:
        check_input_reachability(op, whole, mesh)
    except ValueError as error:
        raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    for param, pattern, type_ in zip(params, tiles.patterns, tiles.operands, strict=True):
        if pattern is not None and not matcher.match(pattern, type_):
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
                frame=frame, current_mesh=mesh, mesh_scope=() if mesh is None else (mesh,)
            )
        )
        try:
            handler(call, ctx)
        except VerifyError as error:
            raise ScheduleRefusal(str(error), path=matcher.bindings, op=op) from error
    if atom is not None:
        boundaries = operand_relations(op, tiles.operands)
        rows = tuple(
            None
            if (row := matched_row_issues(pattern, matcher)) is None
            else (projected_axes(boundary)[row[0]], row[1])
            for pattern, boundary in zip(tiles.patterns, boundaries, strict=False)
        )
        for row in rows:
            if row is None:
                continue
            axis, width = row
            serial = tiles.repeat[axis] // tiles.parallel[axis]
            beside = min(width, serial)
            if serial % beside:
                raise ScheduleRefusal(
                    f"iteration axis {axis} repeat {serial} is not divisible by row issues {beside}",
                    path=matcher.bindings,
                    op=op,
                )
        tiles = replace(tiles, rows=rows)
    return dict(matcher.bindings), tiles
