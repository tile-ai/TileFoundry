"""HIR operation that selects one registered TIR instruction for a tensor tile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, cost, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types


@dataclass(frozen=True)
class InstructionMatch:
    """An accepted constructed instruction and its closed matcher's ordered bindings."""

    op: expr.Op
    path: tuple[tuple[str, object], ...]


@expr.register_op(dialect="tf", category="schedule", name="schedule")
class ScheduleOp(expr.Op):
    """Apply one TIR instruction repeatedly over a tuple of HIR operands."""

    scope_inputs = ()

    operands = expr.ParamDef(
        kind="input", annotation=Tuple[pattern.is_ranked_tensor()], pattern=pattern.is_ranked_tensor()
    )
    op = expr.ParamDef(kind="attribute", annotation=expr.Op)
    repeat = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    order = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)

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


@evaluator.register(ScheduleOp)
def eval_schedule(ctx):
    selected = ctx.op.op
    handler = evaluator.registry.get(type(selected), None)
    if handler is None:
        raise evaluator.EvalError(f"no schedule evaluator registered for {type(selected).__name__}")
    return handler(ctx.for_op(selected, ctx.args, ctx.result_type))


def instruction_schema(
    op: expr.Op, operand_count: int | None = None
) -> tuple[tuple[expr.ParamDef, ...], ...]:
    schema = expr.schema_of(type(op))
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
    writes = tuple(param for param in params if param.effect & expr.MemoryEffect.WRITE)
    if not writes:
        raise ValueError(f"{type(op).__name__} writes no result operand")
    return params, writes


def outer_band(
    relations: tuple[access.Relation, ...],
    repeat: tuple[int, ...],
    order: tuple[int, ...],
    single_shape: tuple[int, ...],
) -> tuple[access.Relation, ...]:
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

    def lifted(acc: access.Relation) -> access.Relation:
        relation = band.apply_range(acc.relation.affine_hull())
        names = (
            relation.get_dim_name(isl.dim_type.PARAM, index)
            for index in range(relation.dim(isl.dim_type.PARAM))
        )
        return access.Relation(
            relation,
            {name: acc.values[name] for name in names if name in acc.values},
            acc.may_access,
        )

    return tuple(lifted(acc) for acc in relations)


@access.register_relation(ScheduleOp)
def schedule_access_relation(call: expr.Call, ctx) -> tuple[access.Relation, ...]:
    sched = call.target
    tiles = schedule.project(
        sched.op,
        tuple(ctx.type_of(arg) for arg in call.args),
        ctx.current_mesh,
        repeat=sched.repeat,
        order=sched.order,
        fragments=False,
    )
    scheduled = outer_band(
        schedule.operand_relations(sched.op, tiles.tiles), tiles.repeat, tiles.order, tiles.shape
    )
    return scheduled


@cost.register(ScheduleOp)
def schedule_cost(call: expr.Call, ctx) -> cost.Cost:
    evaluator_value = cost.registry.get(type(call.target.op), None)
    if evaluator_value is not None:
        return evaluator_value(expr.Call(call.target.op, call.args, type=call.type), ctx)
    stated = access.relations_of(call, ctx)
    local = access.projected(stated, call, ctx)
    type_list = (*(ctx.local_type_of(arg) for arg in call.args), ctx.local_output_type(call))
    params, writes = instruction_schema(call.target.op, len(call.args))
    moved = []
    for index, (type_, boundary) in enumerate(zip(type_list, local, strict=True)):
        if not isinstance(type_, types.TensorType):
            raise ValueError("ScheduleOp cost requires tensor operands and output")
        elements = access.reached_elements(boundary)
        if elements is None:
            raise ValueError(f"ScheduleOp boundary {index} has no finite traffic")
        amount = -(-(elements * type_.dtype.bit_width) // 8)
        moved.append(
            cost.TrafficBytes(
                read=amount
                if index < len(params) and params[index].effect & expr.MemoryEffect.READ
                else 0,
                write=amount
                if index < len(params) and params[index].effect & expr.MemoryEffect.WRITE
                else 0,
            )
        )
    if any(param.effect & expr.MemoryEffect.READ for param in writes):
        iterations = utils.cardinality(access.iteration_universe(local))
        if iterations is None:
            raise ValueError("ScheduleOp has no finite floating-point iteration count")
        dtype = next(
            (
                type_.dtype
                for param, type_ in zip(params, type_list, strict=False)
                if not param.effect & expr.MemoryEffect.WRITE and isinstance(type_, types.TensorType)
            ),
            type_list[-1].dtype,
        )
        return cost.operation(dtype, 2 * iterations, tuple(moved))
    return cost.Cost({}, tuple(moved))


def match_instruction(op_type, attributes, operand_types, mesh, *, frame, repeat=None, order=None):
    """Enumerate atoms, admitting each concrete call through the shared checker."""
    op = op_type(**attributes)
    params, _ = instruction_schema(op, len(operand_types))
    if not all(isinstance(type_, types.TensorType) for type_ in operand_types):
        raise schedule.RefusalError(f"{op_type.__name__} schedule operands must be tensors", op=op)
    atom = getattr(op, "atom", None)
    if not isinstance(atom, type):
        matcher = pattern.Matcher(dict(getattr(atom, "bindings", {})))
        schedule.admit_fields(op, operand_types, mesh, matcher, target=None if frame is None else frame.target)
        tiles = schedule.project(op, operand_types, mesh, bindings=matcher.bindings, repeat=repeat, order=order)
        bindings, _ = schedule.check_call(
            op, operand_types, mesh, frame=frame, repeat=repeat, order=order, projection=tiles
        )
        return (InstructionMatch(op, tuple(bindings.items())),)
    atom_param = next(param for param in expr.schema_of(op_type).signature if param.name == "atom")
    if atom not in atom_param.pattern.declarations:
        raise schedule.RefusalError(
            f"{atom.__name__} is not a declared atom of {op_type.__name__}", op=op
        )
    largest = max(
        (extent for type_ in operand_types for extent in type_.shape if type(extent) is int),
        default=1,
    )
    integers = tuple(param for param in atom.parameters if param.annotation is int)
    partial = pattern.variants(
        tuple(param for param in atom.parameters if param.annotation is not int),
        vary_defaulted=True,
    )
    states = []
    refusal = None
    for seed in partial:
        matcher = pattern.Matcher(seed)
        try:
            schedule.admit_fields(op, operand_types, mesh, matcher, target=None if frame is None else frame.target)
        except schedule.RefusalError as error:
            refusal = refusal or error
            continue
        held = [seed]
        for param in integers:
            axes = pattern.declared_symbol_axes(params, op, matcher.bindings, param.name)
            held = [
                {**state, param.name: value}
                for state in held
                for value in range(1, largest + 1)
                if all(
                    type(extent := operand_types[index].shape[axis]) is not int
                    or extent % value == 0
                    for index, axis in axes
                )
                and (probe := pattern.Matcher(state)).match(param.pattern, value)
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
            tiles = schedule.project(instance, operand_types, mesh, repeat=repeat, order=order)
            bindings, _ = schedule.check_call(
                instance,
                operand_types,
                mesh,
                frame=frame,
                repeat=repeat,
                order=order,
                projection=tiles,
            )
        except schedule.RefusalError as error:
            refusal = refusal or error
        else:
            accepted.append(InstructionMatch(instance, tuple(bindings.items())))
    if not accepted:
        raise refusal or schedule.RefusalError(
            f"{expr.get_registered_name(atom)} has no admitted tile parameters", op=op
        )
    return tuple(accepted)


@typeinfer.register(ScheduleOp)
def infer_schedule(call: expr.Call, ctx) -> types.TensorType:
    op = call.target.op
    if not all(isinstance(ctx.type_of(arg), types.TensorType) for arg in call.args):
        ctx.error(call, f"{type(op).__name__} schedule operands must be tensors")
    try:
        if isinstance(getattr(op, "atom", None), type):
            attributes = {
                param.name: getattr(op, param.name)
                for param in expr.schema_of(type(op)).signature
                if param.kind == "attribute"
            }
            match_instruction(
                type(op),
                attributes,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                frame=ctx,
                repeat=call.target.repeat,
                order=call.target.order,
            )
        else:
            schedule.admit_fields(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                pattern.Matcher(dict(getattr(getattr(op, "atom", None), "bindings", {}))),
                target=ctx.target,
            )
            tiles = schedule.project(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                repeat=call.target.repeat,
                order=call.target.order,
            )
            schedule.check_call(
                op,
                tuple(ctx.type_of(arg) for arg in call.args),
                ctx.current_mesh,
                frame=ctx,
                repeat=call.target.repeat,
                order=call.target.order,
                projection=tiles,
            )
    except schedule.RefusalError as error:
        ctx.error(call, str(error))
    params, writes = instruction_schema(op, len(call.args))
    return ctx.type_of(call.args[params.index(writes[0])])


@access.register_alias(ScheduleOp)
def schedule_result_operand(call: expr.Call) -> int:
    params, writes = instruction_schema(call.target.op, len(call.args))
    return params.index(writes[0])


__all__ = [
    "ScheduleOp",
    "InstructionMatch",
    "instruction_schema",
    "match_instruction",
]
