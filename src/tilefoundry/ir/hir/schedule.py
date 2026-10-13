"""HIR operation that selects one registered TIR instruction for a tensor tile."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import isl

from tilefoundry.evaluator.registry import register_eval, schedule_eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Op, Var, op_identifier
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    PatternMatcher,
    ShardLayoutPattern,
    TensorPattern,
    is_ranked_tensor,
)
from tilefoundry.ir.pattern.utils import (
    declared_execution_mesh,
    declared_shape,
    selected_pattern,
    variants,
)
from tilefoundry.ir.types import TensorType, UnitType
from tilefoundry.ir.types.shard_layout import shard_layout_of, split_target_axes
from tilefoundry.ir.types.utils import tile_inner_type
from tilefoundry.utils.isl_utils import cardinality, involved_dims
from tilefoundry.visitor_registry import (
    register_cost_evaluator,
    register_typeinfer,
    verify_stmt_registry,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    iteration_universe,
    projected,
    projected_axes,
    reached_elements,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias
from tilefoundry.visitor_registry.contexts import (
    Cost,
    TrafficBytes,
    TypeInferContext,
    VerifyContext,
)
from tilefoundry.visitor_registry.cost_utils import operation_cost
from tilefoundry.visitor_registry.verify import verify_between


@dataclass(frozen=True)
class InstructionMatch:
    """An accepted constructed instruction and its closed matcher's ordered bindings."""

    op: Op
    path: tuple[tuple[str, object], ...]


class ScheduleRefusal(ValueError):
    """An instruction's failed node and the bindings reached before that refusal."""

    def __init__(self, reason, *, path=(), instruction=""):
        self.reason = reason
        self.path = tuple(path)
        self.instruction = instruction
        super().__init__(reason)

    def __str__(self):
        from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

        printer = PatternPrinter()
        return printer.match_text(self.instruction, printer.match_data((self.path,), self.reason)).lstrip()


@register_op(dialect="tf", category="schedule", name="schedule")
class ScheduleOp(Op):
    """Apply one TIR instruction repeatedly over a tuple of HIR operands."""

    operands = ParamDef(kind="input", annotation=Tuple[is_ranked_tensor()], pattern=is_ranked_tensor())
    op = ParamDef(kind="attribute", annotation=Op)
    repeat = ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    order = ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)

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


@register_eval(ScheduleOp)
def _eval_schedule(ctx):
    selected = ctx.op.op
    handler = schedule_eval_registry.lookup(type(selected))
    if handler is None:
        raise EvalError(f"no schedule evaluator registered for {type(selected).__name__}")
    return handler(ctx.for_op(selected, ctx.args, ctx.result_type))


class _RelationContext:
    def __init__(self, types: dict[int, TensorType]) -> None:
        self._types = types

    def type_of(self, expr: Var) -> TensorType:
        return self._types[id(expr)]


def operand_relations(
    op: Op, operand_types: tuple[TensorType, ...]
) -> tuple[AccessRelation, ...]:
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


def instruction_schema(
    op: Op, operand_count: int | None = None
) -> tuple[tuple[ParamDef, ...], ...]:
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        raise ValueError(f"{type(op).__name__} is not a registered operation")
    params = tuple(param for param in schema.signature if param.kind == "input")
    if any(param.effect is None for param in params):
        raise ValueError(f"{type(op).__name__} does not declare every operand's memory effect")
    if operand_count is not None:
        required = sum(not param.optional for param in params)
        optional = tuple(param for param in params if param.optional)
        supplied_optional = operand_count - required
        included = {id(param) for param in optional[: max(0, supplied_optional)]}
        params = tuple(param for param in params if not param.optional or id(param) in included)
    writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    if not writes:
        raise ValueError(f"{type(op).__name__} writes no result operand")
    return params, writes


def _instruction_view(call: Call, ctx, *, fragments: bool = True, bindings=None):
    schedule = call.target
    op = schedule.op
    params, writes = instruction_schema(op, len(call.args))
    bindings = (
        dict(getattr(getattr(op, "atom", None), "bindings", {}))
        if bindings is None
        else dict(bindings)
    )
    patterns = tuple(_operand_pattern(param, op, bindings) for param in params)
    whole = {param.name: ctx.type_of(arg) for param, arg in zip(params, call.args, strict=True)}
    try:
        whole_relations = operand_relations(op, tuple(whole[param.name] for param in params))
    except ValueError as error:
        raise ScheduleRefusal(str(error)) from error
    whole_shape = _iteration_shape(whole_relations)
    operands = whole_relations[: len(params)]
    write_boundaries = tuple(
        boundary
        for param, boundary in zip(params, operands, strict=True)
        if param.effect & MemoryEffect.WRITE
    )
    collapsed = frozenset(range(len(whole_shape))) - set().union(
        *(involved_dims(boundary.relation) for boundary in write_boundaries)
    )
    if fragments and getattr(op, "atom", None) is None:
        for type_, boundary in zip(whole.values(), operands, strict=True):
            layout = shard_layout_of(type_.layout)
            if layout is None:
                continue
            axes = projected_axes(boundary)
            if any(
                axis is not None and axes[axis] in collapsed
                for axis in split_target_axes(layout, type_.shape)
            ):
                fragments = False
                break
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
        raise ScheduleRefusal(str(error)) from error
    single_shape = _iteration_shape(single)
    if len(whole_shape) != len(single_shape):
        raise ScheduleRefusal("single-issue and scheduled iteration ranks differ")
    inferred = []
    for whole_extent, single_extent in zip(whole_shape, single_shape, strict=True):
        if single_extent < 1 or whole_extent % single_extent:
            raise ScheduleRefusal(
                f"iteration extent {whole_extent} is not divisible by "
                f"single-issue extent {single_extent}"
            )
        inferred.append(whole_extent // single_extent)
    inferred = tuple(inferred)
    repeat = inferred if schedule.repeat is None else schedule.repeat
    fixed = any(declared_shape(pattern, bindings) is not None for pattern in patterns)
    if not fixed and any(count != 1 for count in repeat):
        raise ScheduleRefusal("transfer tiling is not yet supported")
    if repeat != inferred:
        raise ScheduleRefusal(f"repeat {repeat} conflicts with inferred repeat {inferred}")
    order = tuple(range(len(repeat))) if schedule.order is None else schedule.order
    if sorted(order) != list(range(len(repeat))):
        raise ScheduleRefusal(f"order {order} is not a permutation of iteration dimensions")
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
                enclosing=ctx.current_mesh,
                shard_attrs=shard_attrs,
            )
        except ValueError as error:
            raise ScheduleRefusal(str(error)) from error
        inner_types.append(inner)
    inner_types = tuple(inner_types)
    return op, params, writes, patterns, inner_types, repeat, order, single_shape


def _outer_band(
    relations: tuple[AccessRelation, ...],
    repeat: tuple[int, ...],
    order: tuple[int, ...],
    single_shape: tuple[int, ...],
) -> tuple[AccessRelation, ...]:
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

    def lifted(access: AccessRelation) -> AccessRelation:
        relation = band.apply_range(access.relation.affine_hull())
        names = (
            relation.get_dim_name(isl.dim_type.PARAM, index)
            for index in range(relation.dim(isl.dim_type.PARAM))
        )
        return AccessRelation(
            relation,
            {name: access.values[name] for name in names if name in access.values},
            access.may_access,
        )

    return tuple(lifted(access) for access in relations)


@register_access_relation(ScheduleOp)
def _schedule_access_relation(call: Call, ctx) -> tuple[AccessRelation, ...]:
    op, params, _writes, _patterns, inner, repeat, order, shape = _instruction_view(
        call, ctx, fragments=False
    )
    scheduled = _outer_band(operand_relations(op, inner), repeat, order, shape)
    return scheduled


@register_cost_evaluator(ScheduleOp)
def _schedule_cost(call: Call, ctx) -> Cost:
    stated = relations_of(call, ctx)
    local = projected(stated, call, ctx)
    types = (*(ctx.local_type_of(arg) for arg in call.args), ctx.local_output_type(call))
    params, writes = instruction_schema(call.target.op, len(call.args))
    moved = []
    for index, (type_, boundary) in enumerate(zip(types, local, strict=True)):
        if not isinstance(type_, TensorType):
            raise ValueError("ScheduleOp cost requires tensor operands and output")
        elements = reached_elements(boundary)
        if elements is None:
            raise ValueError(f"ScheduleOp boundary {index} has no finite traffic")
        amount = -(-(elements * type_.dtype.bit_width) // 8)
        moved.append(
            TrafficBytes(
                read=amount
                if index < len(params) and params[index].effect & MemoryEffect.READ
                else 0,
                write=amount
                if index < len(params) and params[index].effect & MemoryEffect.WRITE
                else 0,
            )
        )
    if any(param.effect & MemoryEffect.READ for param in writes):
        iterations = cardinality(iteration_universe(local))
        if iterations is None:
            raise ValueError("ScheduleOp has no finite floating-point iteration count")
        dtype = next(
            (
                type_.dtype
                for param, type_ in zip(params, types, strict=False)
                if not param.effect & MemoryEffect.WRITE and isinstance(type_, TensorType)
            ),
            types[-1].dtype,
        )
        return operation_cost(dtype, 2 * iterations, tuple(moved))
    return Cost({}, tuple(moved))


def _pattern_refusal(matcher: PatternMatcher, parameter=None) -> ScheduleRefusal:
    from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

    refusal = matcher.refusal
    path = tuple((refusal.bindings if refusal is not None else matcher.bindings).items())
    return ScheduleRefusal(PatternPrinter().refusal_reason(refusal, parameter), path=path)


def _instruction_call(op, operand_types, mesh, *, repeat, order, scope):
    args = tuple(
        Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(operand_types)
    )
    call = Call(ScheduleOp(op=op, repeat=repeat, order=order), args, type=UnitType())
    ctx = TypeInferContext(
        scope=scope,
        current_mesh=mesh,
        memo={id(arg): (arg, type_) for arg, type_ in zip(args, operand_types, strict=True)},
    )
    return call, ctx


def _check_instruction(op, operand_types, mesh, *, repeat, order, scope, bindings):
    call, ctx = _instruction_call(op, operand_types, mesh, repeat=repeat, order=order, scope=scope)
    op, params, _writes, patterns, inner, _repeat, _order, _shape = _instruction_view(call, ctx)
    handler = verify_stmt_registry.lookup(type(op))
    if handler is None:
        raise ScheduleRefusal(f"{type(op).__name__} has no registered verifier")
    matcher = PatternMatcher({**bindings, **dict(getattr(getattr(op, "atom", None), "bindings", {}))})
    for param, pattern, type_ in zip(params, patterns, inner, strict=True):
        if pattern is not None and not matcher.match(pattern, type_):
            raise _pattern_refusal(matcher, param.name)
    if not matcher.solve():
        raise _pattern_refusal(matcher)
    args = tuple(
        Var(name=param.name, type=type_) for param, type_ in zip(params, inner, strict=True)
    )
    target_call = Call(target=op, args=args, type=UnitType())
    frames = tuple(layout.mesh for type_ in inner if hasattr((layout := type_.layout), "mesh"))
    verify_ctx = VerifyContext(
        scope=scope,
        current_mesh=mesh,
        memo={id(arg): (arg, type_) for arg, type_ in zip(args, inner, strict=True)},
        mesh_scope=frames[:1] or (() if mesh is None else (mesh,)),
    )
    try:
        handler(target_call, verify_ctx)
        verify_between(target_call, verify_ctx)
    except VerifyError as error:
        raise ScheduleRefusal(str(error), path=tuple(matcher.bindings.items())) from error
    return InstructionMatch(op, tuple(matcher.bindings.items()))


def _match_instruction(
    op_type, attributes, operand_types, mesh, *, scope, repeat=None, order=None
) -> tuple[InstructionMatch, ...]:
    """Return every instruction instance implementing these types and execution scope.

    Read layout forks for each projectable tile, then validate the constructed
    instance with the same closed-instruction checks.
    """
    op = op_type(**attributes)
    params, _writes = instruction_schema(op, len(operand_types))
    if len(params) != len(operand_types):
        raise ScheduleRefusal(
            f"{op_type.__name__} expects operands {tuple(param.name for param in params)}, got {len(operand_types)}"
        )
    if not all(isinstance(type_, TensorType) for type_ in operand_types):
        raise ScheduleRefusal(f"{op_type.__name__} schedule operands must be tensors")
    atom = getattr(op, "atom", None)
    initial = {} if isinstance(atom, type) else dict(getattr(atom, "bindings", {}))
    matcher = PatternMatcher(initial)
    if not matcher.match(declared_execution_mesh(op), mesh):
        raise _pattern_refusal(matcher, "mesh")
    from tilefoundry.visitor_registry.typeinfer import input_scope_refusal  # noqa: PLC0415

    refusal = input_scope_refusal(operand_types, mesh)
    if refusal is not None:
        raise ScheduleRefusal(refusal, path=tuple(matcher.bindings.items()))
    initial = dict(matcher.bindings)
    patterns = tuple(_operand_pattern(param, op, initial) for param in params)
    for field in ("dtype", "storage"):
        matcher = PatternMatcher(initial)
        for param, pattern, type_ in zip(params, patterns, operand_types, strict=True):
            if pattern is None:
                continue
            fields = replace(
                pattern, shape=None, layout=None, predicates=(),
                storage=None if field == "dtype" else pattern.storage,
            )
            if not matcher.match(fields, type_):
                raise _pattern_refusal(matcher, param.name)
    if not isinstance(atom, type):
        try:
            return (_check_instruction(op, operand_types, mesh, repeat=repeat, order=order, scope=scope, bindings=initial),)
        except ScheduleRefusal as error:
            if not error.path:
                error.path = tuple(matcher.bindings.items())
            raise
    atom_param = next(param for param in op_type._op_schema.signature if param.name == "atom")
    if atom not in atom_param.pattern.declarations:
        raise ScheduleRefusal(f"{atom.__name__} is not a declared atom of {op_type.__name__}")
    bindings = dict(matcher.bindings)
    integers = tuple(
        param for param in atom.parameters if param.annotation is int and param.name not in bindings
    )
    largest = max(
        (extent for type_ in operand_types for extent in type_.shape if type(extent) is int),
        default=1,
    )
    choices = variants(
        integers, vary_defaulted=True, values=lambda param: tuple(range(1, largest + 1))
    )
    choices = tuple(sorted(choices, key=lambda state: tuple(state.values()), reverse=True))
    call, ctx = _instruction_call(op, operand_types, mesh, repeat=repeat, order=order, scope=scope)
    first_refusal = None
    tile_refusal = None
    divisible_seen = False
    tile_choices = []
    for choice in choices:
        bound = {**bindings, **choice}
        try:
            _instruction_view(call, ctx, fragments=False, bindings=bound)
        except ScheduleRefusal as error:
            if not error.path:
                error.path = tuple(bound.items())
            first_refusal = first_refusal or error
            continue
        try:
            view = _instruction_view(call, ctx, bindings=bound)
        except ScheduleRefusal as error:
            if not error.path:
                error.path = tuple(bound.items())
            if not divisible_seen:
                tile_refusal = error
            divisible_seen = True
            continue
        divisible_seen = True
        tile_choices.append((choice, view))
    if not tile_choices:
        raise (
            tile_refusal
            or first_refusal
            or ScheduleRefusal(f"{atom.reference_name} has no admitted tile parameters")
        )
    parameters = {param.name for param in atom.parameters}
    integer_names = {param.name for param in integers}
    accepted = []
    refusal = None
    for choice, (_op, _params, _writes, patterns, inner, _repeat, _order, _shape) in tile_choices:
        matcher = PatternMatcher({**bindings, **choice})
        try:
            for param, pattern, type_ in zip(params, patterns, inner, strict=True):
                if (
                    pattern is not None and pattern.layout is not None
                    and not matcher.match(pattern.layout, type_.layout)
                ):
                    raise _pattern_refusal(matcher, param.name)
            if not matcher.solve():
                raise _pattern_refusal(matcher)
            solved = {
                name: value for name, value in matcher.bindings.items()
                if name in parameters and name not in integer_names
            }
            instance = op_type(**{**attributes, "atom": atom(**{**solved, **choice})})
            matched = _check_instruction(
                instance, operand_types, mesh, repeat=repeat, order=order, scope=scope, bindings=initial
            )
        except ScheduleRefusal as error:
            if not error.path:
                error.path = tuple(matcher.bindings.items())
            refusal = refusal or error
        else:
            accepted.append(matched)
    if not accepted:
        raise refusal
    return tuple(accepted)


def match_instruction(op_type, attributes, operand_types, mesh, *, scope, repeat=None, order=None) -> tuple[InstructionMatch, ...]:
    """Match all constructed instructions, preserving their full diagnostic paths."""
    try:
        return _match_instruction(op_type, attributes, operand_types, mesh,
                                  scope=scope, repeat=repeat, order=order)
    except ScheduleRefusal as error:
        atom = attributes.get("atom")
        declaration = atom if isinstance(atom, type) else type(atom) if atom is not None else op_type
        error.instruction = op_identifier(declaration)
        raise


@register_typeinfer(ScheduleOp)
def _infer_schedule(call: Call, ctx) -> TensorType:
    op = call.target.op
    attributes = {
        param.name: getattr(op, param.name)
        for param in type(op)._op_schema.signature
        if param.kind == "attribute"
    }
    try:
        match_instruction(
            type(op),
            attributes,
            tuple(ctx.type_of(arg) for arg in call.args),
            ctx.current_mesh,
            scope=ctx.scope,
            repeat=call.target.repeat,
            order=call.target.order,
        )
    except ScheduleRefusal as error:
        ctx.error(call, str(error))
    params, writes = instruction_schema(op, len(call.args))
    return ctx.type_of(call.args[params.index(writes[0])])


@register_buffer_alias(ScheduleOp)
def _schedule_result_operand(call: Call) -> int:
    params, writes = instruction_schema(call.target.op, len(call.args))
    return params.index(writes[0])


__all__ = ["ScheduleOp", "ScheduleRefusal", "InstructionMatch", "instruction_schema", "match_instruction", "operand_relations"]
