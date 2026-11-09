"""HIR ``Transpose`` Op: permute the axes of a tensor."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='tensor')
class Transpose(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    perm = expr.ParamDef(kind="attribute", annotation=tuple)


@access.register_relation(Transpose)
def transpose_relations(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Result axis k is source axis perm[k], stated in both sides' positions.

    A permutation walks what it reads, so the source's own axes are the
    coordinates and the permutation happens on the way out. Which positions those
    axes are is the reader's question, asked of every Op the same way.
    """
    perm = tuple(call.target.perm)
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    writes_at = [f"d{source_axis}" for source_axis in perm]
    domain = ", ".join(f"d{index}" for index in range(rank))
    return access.iterating(
        source.shape,
        (
            access.identity(rank),
            access.Relation(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}")),
        ),
    )


@typeinfer.register(Transpose)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    """A new compact value with its source axes in another order."""
    x_ty = ctx.type_of(call.args[0])
    perm = call.target.perm
    if len(perm) != len(x_ty.shape):
        ctx.error(call, f"perm length {len(perm)} != rank {len(x_ty.shape)}")
    new_shape = tuple(x_ty.shape[p] for p in perm)

    new_layout = typeinfer.derive_output_shard_layout(
        (x_ty,),
        access.relations_of(call, ctx),
        new_shape,
        fresh_strides=True,
        ctx=ctx,
        storage=x_ty.storage,
    )
    if new_layout is None:
        new_layout = types.Layout(new_shape)
    return types.TensorType(shape=new_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@evaluator.register(Transpose)
def eval_transpose(ctx):
    def transpose(x):
        return utils.torch.permute(x, tuple(ctx.op.perm)).contiguous()

    x = ctx.args[0].tensor
    held = transpose(x.tensor) if x.is_global else tuple((p, transpose(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(Transpose)
def transpose_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    moved = ir_utils.tensor_bytes(cost.output_type(call, ctx))
    return cost.Cost({}, (cost.TrafficBytes(read=moved), cost.TrafficBytes(write=moved)))


__all__ = ["Transpose"]
