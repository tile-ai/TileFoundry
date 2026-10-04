"""HIR operation that selects one registered TIR instruction for a tensor tile."""

from __future__ import annotations

from typing import Tuple

import isl

from tilefoundry.evaluator.registry import register_eval, schedule_eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Op, Var
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    PatternMatcher,
    ShardLayoutPattern,
    TensorPattern,
    is_ranked_tensor,
)
from tilefoundry.ir.pattern.utils import (
    declared_shape,
    declared_write_type,
    selected_pattern,
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
    AccessRelations,
    BoundaryRelation,
    iteration_universe,
    projected,
    projected_axes,
    reached_elements,
    register_access_relation,
    relation_of,
    relations_of,
    restricted_access,
)
from tilefoundry.visitor_registry.contexts import Cost, TrafficBytes, VerifyContext
from tilefoundry.visitor_registry.verify import verify_between


@register_op(dialect="tf", category="schedule", name="schedule")
class ScheduleOp(Op):
    """Apply one TIR instruction repeatedly over a tuple of HIR operands."""

    operands = ParamDef(kind="input", annotation=Tuple[is_ranked_tensor()], pattern=is_ranked_tensor())
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


def _single_issue_relations(op: Op, operand_types: tuple[TensorType, ...]) -> AccessRelations:
    """Ask the selected instruction's registry entry about one issue."""
    args = tuple(
        Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(operand_types)
    )
    call = Call(target=op, args=args, type=UnitType())
    return relations_of(call, _RelationContext(dict(zip(map(id, args), operand_types))))


def _operand_pattern(param: ParamDef, op: Op, bindings: dict) -> TensorPattern | None:
    pattern = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
    selected = selected_pattern(pattern, bindings)
    return selected if isinstance(selected, TensorPattern) else None


def _declared_shape(pattern: TensorPattern | None, bindings: dict) -> tuple | None:
    return declared_shape(pattern, bindings)


def _iteration_shape(relations: AccessRelations) -> tuple[int, ...]:
    space = iteration_universe(relations)
    if space is None:
        raise ValueError("instruction access relations state no iteration space")
    return tuple(
        int(space.dim_max(axis).max_val().num_si()) + 1 for axis in range(space.tuple_dim())
    )


def _instruction_schema(
    op: Op, operand_count: int | None = None
) -> tuple[tuple[ParamDef, ...], ...]:
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        raise ValueError(f"{type(op).__name__} is not a registered operation")
    params = tuple(param for param in schema.signature if param.kind == "input")
    if any(param.effect is None for param in params):
        raise ValueError(f"{type(op).__name__} does not declare every operand's memory effect")
    if operand_count is not None:
        required_reads = sum(
            bool(param.effect & MemoryEffect.READ)
            for param in params
            if not param.optional
        )
        optional_reads = tuple(
            param for param in params if param.optional and param.effect & MemoryEffect.READ
        )
        supplied_optional = operand_count - required_reads
        included = {id(param) for param in optional_reads[: max(0, supplied_optional)]}
        params = tuple(
            param for param in params if not param.optional or id(param) in included
        )
    reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
    writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    if not writes:
        raise ValueError(f"{type(op).__name__} writes no result operand")
    return params, reads, writes


def _instruction_view(call: Call, ctx, *, fragments: bool = True):
    schedule = call.target
    op = schedule.op
    params, reads, writes = _instruction_schema(op, len(call.args))
    if len(call.args) != len(reads):
        raise ValueError(f"{type(op).__name__} reads {len(reads)} operands, got {len(call.args)}")
    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    patterns = tuple(_operand_pattern(param, op, bindings) for param in params)
    whole = {param.name: ctx.type_of(arg) for param, arg in zip(reads, call.args, strict=True)}
    if not all(isinstance(type_, TensorType) for type_ in whole.values()):
        raise ValueError(f"{type(op).__name__} schedule operands must be tensors")
    for param, pattern in zip(params, patterns, strict=True):
        if param.effect == MemoryEffect.WRITE and pattern is None:
            raise ValueError(
                f"{type(op).__name__} {param.name} is write-only and declares no result shape"
            )
    whole_relations = _single_issue_relations(
        op, tuple(whole.get(param.name, UnitType()) for param in params)
    )
    whole_shape = _iteration_shape(whole_relations)
    read_boundaries = tuple(
        boundary
        for param, boundary in zip(params, whole_relations.inputs, strict=True)
        if param.effect & MemoryEffect.READ
    )
    write_boundaries = tuple(
        boundary
        for param, boundary in zip(params, whole_relations.inputs, strict=True)
        if param.effect & MemoryEffect.WRITE
    )
    collapsed = frozenset(range(len(whole_shape))) - set().union(
        *(involved_dims(boundary.pattern.relation) for boundary in write_boundaries)
    )
    if fragments and getattr(op, "atom", None) is None:
        for type_, boundary in zip(whole.values(), read_boundaries, strict=True):
            layout = shard_layout_of(type_.layout)
            if layout is None:
                continue
            axes = projected_axes(boundary.pattern)
            if any(
                axis is not None and axes[axis] in collapsed
                for axis in split_target_axes(layout, type_.shape)
            ):
                fragments = False
                break
    outputs = {}
    for param, pattern in zip(params, patterns, strict=True):
        if param.effect & MemoryEffect.WRITE and not param.effect & MemoryEffect.READ:
            outputs[param.name] = declared_write_type(
                op,
                param,
                pattern,
                whole,
                ctx.current_mesh,
                AccessRelations(read_boundaries, (write_boundaries[writes.index(param)],)),
                collapsed,
            )
    whole.update(outputs)
    whole_types = tuple(whole[param.name] for param in params)
    shapes = tuple(
        _declared_shape(pattern, bindings) or tuple(type_.shape)
        for pattern, type_ in zip(patterns, whole_types, strict=True)
    )
    provisional = tuple(
        TensorType(shape, type_.dtype, type_.layout, type_.storage)
        for shape, type_ in zip(shapes, whole_types, strict=True)
    )
    single = _single_issue_relations(op, provisional)
    single_shape = _iteration_shape(single)
    if len(whole_shape) != len(single_shape):
        raise ValueError("single-issue and scheduled iteration ranks differ")
    inferred = []
    for whole_extent, single_extent in zip(whole_shape, single_shape, strict=True):
        if single_extent < 1 or whole_extent % single_extent:
            raise ValueError(
                f"iteration extent {whole_extent} is not divisible by "
                f"single-issue extent {single_extent}"
            )
        inferred.append(whole_extent // single_extent)
    inferred = tuple(inferred)
    repeat = inferred if schedule.repeat is None else schedule.repeat
    fixed = any(_declared_shape(pattern, bindings) is not None for pattern in patterns)
    if not fixed and any(count != 1 for count in repeat):
        raise ValueError("transfer tiling is not yet supported")
    if repeat != inferred:
        raise ValueError(f"repeat {repeat} conflicts with inferred repeat {inferred}")
    order = tuple(range(len(repeat))) if schedule.order is None else schedule.order
    if sorted(order) != list(range(len(repeat))):
        raise ValueError(f"order {order} is not a permutation of iteration dimensions")
    participant = getattr(getattr(op, "atom", None), "required_execution_mesh", None)
    read_args = dict(zip((param.name for param in reads), call.args, strict=True))
    inner_types = (
        tuple(
            tile_inner_type(
                (
                    TensorType(
                        type_.shape,
                        type_.dtype,
                        ctx.type_of(read_args[param.name]).layout,
                        type_.storage,
                    )
                    if param.name in read_args
                    else type_
                ),
                shape,
                counts=tuple(
                    1 if axis is None else repeat[axis]
                    for axis in projected_axes(boundary.pattern)
                ),
                participant=participant,
                enclosing=ctx.current_mesh,
                shard_attrs=(
                    pattern.layout.attrs
                    if pattern is not None and isinstance(pattern.layout, ShardLayoutPattern)
                    else None
                ),
            )
            for param, type_, shape, pattern, boundary in zip(
                params, whole_types, shapes, patterns, single.inputs, strict=True
            )
        )
        if fragments
        else provisional
    )
    return op, params, reads, writes, patterns, inner_types, repeat, order, single_shape


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
        return BoundaryRelation(restricted_access(relation, pattern.values))

    return AccessRelations(
        inputs=tuple(lifted(boundary) for boundary in relations.inputs),
        outputs=tuple(lifted(boundary) for boundary in relations.outputs),
    )


@register_access_relation(ScheduleOp)
def _schedule_access_relation(call: Call, ctx) -> AccessRelations:
    op, params, _reads, _writes, _patterns, inner, repeat, order, shape = _instruction_view(
        call, ctx, fragments=False
    )
    scheduled = _outer_band(_single_issue_relations(op, inner), repeat, order, shape)
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
    _params, reads, writes = _instruction_schema(op, len(call.args))
    flops = {}
    if any(param.effect & MemoryEffect.READ for param in writes):
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


@register_typeinfer(ScheduleOp)
def _infer_schedule(call: Call, ctx) -> TensorType:
    try:
        op, params, reads, writes, patterns, inner, _repeat, _order, _shape = _instruction_view(
            call, ctx
        )
    except (TypeError, ValueError) as error:
        ctx.error(call, str(error))
    handler = verify_stmt_registry.lookup(type(op))
    if handler is None:
        ctx.error(call, f"{type(op).__name__} has no registered verifier")
    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    matcher = PatternMatcher(bindings)
    for param, pattern, type_ in zip(params, patterns, inner, strict=True):
        if pattern is None or matcher.match(pattern, type_):
            continue
        from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

        ctx.error(
            call,
            f"{type(op).__name__} {param.name} single-issue tile does not match: "
            f"{PatternPrinter().refusal(matcher.refusal)}",
        )
    if not matcher.solve():
        from tilefoundry.inspection.pattern_printer import PatternPrinter  # noqa: PLC0415

        ctx.error(call, PatternPrinter().refusal(matcher.refusal))
    args = tuple(
        Var(name=param.name, type=type_) for param, type_ in zip(params, inner, strict=True)
    )
    target_call = Call(target=op, args=args, type=UnitType())
    frames = tuple(layout.mesh for type_ in inner if hasattr((layout := type_.layout), "mesh"))
    verify_ctx = VerifyContext(
        scope=ctx.scope,
        current_mesh=ctx.current_mesh,
        memo={id(arg): (arg, type_) for arg, type_ in zip(args, inner, strict=True)},
        mesh_scope=frames[:1] or (() if ctx.current_mesh is None else (ctx.current_mesh,)),
    )
    handler(target_call, verify_ctx)
    verify_between(target_call, verify_ctx)
    result = writes[0]
    if result.effect & MemoryEffect.READ:
        return ctx.type_of(call.args[reads.index(result)])
    return inner[params.index(result)]


__all__ = ["ScheduleOp"]
