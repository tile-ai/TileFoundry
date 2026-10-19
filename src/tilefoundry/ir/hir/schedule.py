"""HIR operation that selects one registered TIR instruction for a tensor tile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import isl

from tilefoundry.evaluator.registry import register_eval, schedule_eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    PatternMatcher,
    is_ranked_tensor,
)
from tilefoundry.ir.pattern.utils import (
    declared_symbol_axes,
    variants,
)
from tilefoundry.ir.types import TensorType
from tilefoundry.utils.isl_utils import cardinality
from tilefoundry.visitor_registry import (
    register_cost_evaluator,
    register_typeinfer,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    iteration_universe,
    projected,
    reached_elements,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias
from tilefoundry.visitor_registry.contexts import (
    Cost,
    TrafficBytes,
)
from tilefoundry.visitor_registry.cost_utils import operation_cost
from tilefoundry.visitor_registry.instruction import (
    ScheduleRefusal,
    admit_fields,
    check_call,
    operand_relations,
    project,
)
from tilefoundry.visitor_registry.registries import cost_evaluator_registry


@dataclass(frozen=True)
class InstructionMatch:
    """An accepted constructed instruction and its closed matcher's ordered bindings."""

    op: Op
    path: tuple[tuple[str, object], ...]


@register_op(dialect="tf", category="schedule", name="schedule")
class ScheduleOp(Op):
    """Apply one TIR instruction repeatedly over a tuple of HIR operands."""

    scope_inputs = ()

    operands = ParamDef(
        kind="input", annotation=Tuple[is_ranked_tensor()], pattern=is_ranked_tensor()
    )
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
    schedule = call.target
    tiles = project(
        schedule.op,
        tuple(ctx.type_of(arg) for arg in call.args),
        ctx.current_mesh,
        repeat=schedule.repeat,
        order=schedule.order,
        fragments=False,
    )
    scheduled = _outer_band(
        operand_relations(schedule.op, tiles.operands), tiles.repeat, tiles.order, tiles.shape
    )
    return scheduled


@register_cost_evaluator(ScheduleOp)
def _schedule_cost(call: Call, ctx) -> Cost:
    evaluator = cost_evaluator_registry.lookup(type(call.target.op))
    if evaluator is not None:
        return evaluator(Call(call.target.op, call.args, type=call.type), ctx)
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


def match_instruction(op_type, attributes, operand_types, mesh, *, frame, repeat=None, order=None):
    """Enumerate atoms, admitting each concrete call through the shared checker."""
    op = op_type(**attributes)
    params, _writes = instruction_schema(op, len(operand_types))
    if not all(isinstance(type_, TensorType) for type_ in operand_types):
        raise ScheduleRefusal(f"{op_type.__name__} schedule operands must be tensors", op=op)
    atom = getattr(op, "atom", None)
    if not isinstance(atom, type):
        admit_fields(op, operand_types, mesh, PatternMatcher(dict(getattr(atom, "bindings", {}))))
        tiles = project(op, operand_types, mesh, repeat=repeat, order=order)
        bindings, _tiles = check_call(
            op, operand_types, mesh, frame=frame, repeat=repeat, order=order, projection=tiles
        )
        return (InstructionMatch(op, tuple(bindings.items())),)
    atom_param = next(param for param in op_type._op_schema.signature if param.name == "atom")
    if atom not in atom_param.pattern.declarations:
        raise ScheduleRefusal(
            f"{atom.__name__} is not a declared atom of {op_type.__name__}", op=op
        )
    largest = max(
        (extent for type_ in operand_types for extent in type_.shape if type(extent) is int),
        default=1,
    )
    integers = tuple(param for param in atom.parameters if param.annotation is int)
    partial = variants(
        tuple(param for param in atom.parameters if param.annotation is not int),
        vary_defaulted=True,
    )
    states = []
    refusal = None
    for seed in partial:
        matcher = PatternMatcher(seed)
        try:
            admit_fields(op, operand_types, mesh, matcher)
        except ScheduleRefusal as error:
            refusal = refusal or error
            continue
        held = [seed]
        for param in integers:
            axes = declared_symbol_axes(params, op, matcher.bindings, param.name)
            held = [
                {**state, param.name: value}
                for state in held
                for value in range(1, largest + 1)
                if all(
                    type(extent := operand_types[index].shape[axis]) is not int
                    or extent % value == 0
                    for index, axis in axes
                )
                and (probe := PatternMatcher(state)).match(param.pattern, value)
                and probe.solve()
            ]
        states.extend(held)
    states.sort(key=lambda state: tuple(state[p.name] for p in integers), reverse=True)
    if states:
        refusal = None
    accepted = []
    for state in states:
        instance = op_type(**{**attributes, "atom": atom(**state)})
        try:
            tiles = project(instance, operand_types, mesh, repeat=repeat, order=order)
            bindings, _tiles = check_call(
                instance,
                operand_types,
                mesh,
                frame=frame,
                repeat=repeat,
                order=order,
                projection=tiles,
            )
        except ScheduleRefusal as error:
            refusal = refusal or error
        else:
            accepted.append(InstructionMatch(instance, tuple(bindings.items())))
    if not accepted:
        raise refusal or ScheduleRefusal(
            f"{atom.reference_name} has no admitted tile parameters", op=op
        )
    return tuple(accepted)


@register_typeinfer(ScheduleOp)
def _infer_schedule(call: Call, ctx) -> TensorType:
    op = call.target.op
    if not all(isinstance(ctx.type_of(arg), TensorType) for arg in call.args):
        ctx.error(call, f"{type(op).__name__} schedule operands must be tensors")
    try:
        if isinstance(getattr(op, "atom", None), type):
            attributes = {
                param.name: getattr(op, param.name)
                for param in type(op)._op_schema.signature
                if param.kind == "attribute"
            }
            match_instruction(
                type(op),
                attributes,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                frame=ctx.frame,
                repeat=call.target.repeat,
                order=call.target.order,
            )
        else:
            admit_fields(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                PatternMatcher(dict(getattr(getattr(op, "atom", None), "bindings", {}))),
            )
            tiles = project(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                repeat=call.target.repeat,
                order=call.target.order,
            )
            check_call(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                frame=ctx.frame,
                repeat=call.target.repeat,
                order=call.target.order,
                projection=tiles,
            )
    except ScheduleRefusal as error:
        ctx.error(call, str(error))
    params, writes = instruction_schema(op, len(call.args))
    return ctx.type_of(call.args[params.index(writes[0])])


@register_buffer_alias(ScheduleOp)
def _schedule_result_operand(call: Call) -> int:
    params, writes = instruction_schema(call.target.op, len(call.args))
    return params.index(writes[0])


__all__ = [
    "ScheduleOp",
    "ScheduleRefusal",
    "InstructionMatch",
    "instruction_schema",
    "match_instruction",
    "operand_relations",
]
