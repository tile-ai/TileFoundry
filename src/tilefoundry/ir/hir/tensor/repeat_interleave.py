"""HIR ``RepeatInterleave`` Op: repeat each element along an axis."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(name="repeat_interleave", dialect='tf', category='tensor')
class RepeatInterleave(expr.Op):
    """Repeat each element of ``x`` along ``axis`` ``repeats`` times, interleaved.

    Repeat each element of ``x`` along ``axis`` ``repeats`` times,
    interleaved (GQA head expansion). The named axis grows by ``repeats``;
    all other dims are unchanged.
    """

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    repeats = expr.ParamDef(kind="attribute", annotation=int)
    axis = expr.ParamDef(kind="attribute", annotation=int)


def normalize_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else axis + rank


@typeinfer.register(RepeatInterleave)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    shape = list(x_ty.shape)
    ax = normalize_axis(op.axis, len(shape))
    if not (0 <= ax < len(shape)):
        ctx.error(call, f"RepeatInterleave: axis {op.axis} out of range for rank {len(shape)}")
    shape[ax] = shape[ax] * op.repeats

    new_layout = types.Layout(tuple(shape))
    if isinstance(x_ty.layout, types.ShardLayout) and any(
        isinstance(a, (types.Split, types.Partial)) for a in x_ty.layout.attrs
    ):
        ctx.error(
            call,
            "RepeatInterleave cannot express a sharded layout; reshard to a "
            "replicated layout first",
        )
    return types.TensorType(
        shape=tuple(shape),
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@evaluator.register(RepeatInterleave)
def eval_repeat_interleave(ctx):
    def repeat_interleave(x):
        return utils.torch.repeat_interleave(x, ctx.op.repeats, dim=ctx.op.axis)

    x = ctx.args[0].tensor
    held = repeat_interleave(x.tensor) if x.is_global else tuple((p, repeat_interleave(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@access.register_relation(RepeatInterleave)
def repeat_interleave_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Several result coordinates read one source coordinate, which is read once.

    The pattern is many-to-one and the amount is its image. That three output
    positions depend on one element is the pattern's business; the element still
    crossed the boundary once.
    """
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axis = call.target.axis + rank if call.target.axis < 0 else call.target.axis
    repeats = call.target.repeats
    dims = [f"d{index}" for index in range(rank)]
    reads = list(dims)
    reads[axis] = f"floor(d{axis} / {repeats})" if repeats != 1 else dims[axis]
    domain = ", ".join(dims)
    out_shape = (
        *source.shape[:axis],
        source.shape[axis] * repeats,
        *source.shape[axis + 1 :],
    )
    produced = 1
    for extent in out_shape:
        produced *= extent if isinstance(extent, int) else 1
    return access.iterating(
        out_shape,
        (
            access.Relation(isl.multi_aff(f"{{ [{domain}] -> [{', '.join(reads)}] }}")),
            access.identity(rank),
        ),
    )


@cost.register(RepeatInterleave)
def repeat_interleave_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    inputs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    return cost.Cost({}, cost.traffic(inputs, output))


__all__ = ["RepeatInterleave"]
