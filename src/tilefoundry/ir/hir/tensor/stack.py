"""HIR ``Stack`` Op: join tensors along a new axis."""

from __future__ import annotations

from typing import Tuple

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='tensor')
class Stack(expr.Op):
    """Variadic input op. See Concat for encoding rationale."""

    inputs = expr.ParamDef(kind="input", annotation=Tuple[pattern.is_ranked_tensor()], pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int)


def normalized_axis(call: "Call", ctx: "TypeInferContext", rank: int) -> int:
    raw_axis = call.target.axis
    axis = raw_axis + rank + 1 if raw_axis < 0 else raw_axis
    if not (0 <= axis <= rank):
        ctx.error(call, f"axis {raw_axis} out of range for input rank {rank}")
    return axis


@typeinfer.register(Stack)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    if not call.args:
        ctx.error(call, "Stack requires at least one input")
    type_list = [ctx.type_of(a) for a in call.args]
    base = type_list[0]
    for index, t in enumerate(type_list[1:], start=1):
        if t.shape != base.shape:
            ctx.error(call, f"input {index} shape must match input 0")
        if t.dtype != base.dtype:
            ctx.error(call, f"input {index} dtype must match input 0")
    axis = normalized_axis(call, ctx, len(base.shape))
    new_shape = list(base.shape)
    new_shape.insert(axis, len(call.args))
    new_shape = tuple(new_shape)
    try:
        relation = access.relations_of(call, ctx)
        layout = typeinfer.derive_output_shard_layout(
            tuple(type_list),
            relation,
            new_shape,
            fresh_strides=True,
            ctx=ctx,
            storage=base.storage,
        )
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Stack",
        )
    if layout is not None:
        ir_utils.require_uniform_partial_slices(type_list, layout, "Stack")
    else:
        layout = types.Layout(new_shape)
    storage = typeinfer.resolve_anchor_storage(ctx, call, *(t.storage for t in type_list))
    return types.TensorType(shape=new_shape, dtype=base.dtype, layout=layout, storage=storage)


@evaluator.register(Stack)
def eval_stack(ctx):
    def stack(*xs):
        return utils.torch.stack(xs, dim=ctx.op.axis)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        stack(*(x.tensor for x in xs))
        if not placements
        else tuple((p, stack(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@access.register_relation(Stack)
def stack_access(call: "Call", ctx) -> tuple[access.Relation, ...]:
    """Each input writes one position of the new axis, and is read whole.

    The result has one more axis than its inputs, so an input's relation drops
    that axis rather than pretending to an index on it. Which position an input
    lands at is the guard; how much it moves is its own size. The result keeps
    every axis it has: borrowing an input's map would describe a value one axis
    short, pinned to whichever position that input landed at.
    """
    stacked = tuple(ctx.type_of(call.args[0]).shape)
    rank = len(stacked) + 1
    axis = normalized_axis(call, ctx, rank - 1)
    out_shape = (*stacked[:axis], len(call.args), *stacked[axis:])
    dims = [f"d{index}" for index in range(rank)]
    domain = ", ".join(dims)
    reads = ", ".join(dim for index, dim in enumerate(dims) if index != axis)
    inputs = tuple(
        access.Relation(
            isl.map(f"{{ [{domain}] -> [{reads}] : d{axis} = {position} }}")
            if rank > 1
            else isl.map(f"{{ [{domain}] -> [] : d{axis} = {position} }}")
        )
        for position in range(len(call.args))
    )
    return access.iterating(out_shape, (*inputs, access.identity(rank)))


cost.register(Stack)(cost.structure)


__all__ = ["Stack"]
