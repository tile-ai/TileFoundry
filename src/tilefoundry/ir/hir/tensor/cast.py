"""HIR ``Cast`` Op: convert a tensor to another dtype."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect='tf', category='tensor')
class Cast(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    dtype = expr.ParamDef(kind="attribute", annotation=expr.DType)


access.register_relation(Cast)(access.identity_relations)


@typeinfer.register(Cast)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_layout = x_ty.layout
    if types.shard_layout_of(x_ty.layout) is not None:
        relation = access.relations_of(call, ctx)
        derived = typeinfer.derive_output_shard_layout(
            (x_ty,), relation, x_ty.shape, ctx=ctx, storage=x_ty.storage
        )
        if derived is not None:
            new_layout = derived
    return types.TensorType(
        shape=x_ty.shape, dtype=call.target.dtype, layout=new_layout, storage=x_ty.storage
    )


@evaluator.register(Cast)
def eval_cast(ctx):
    def cast(x):
        return x.to(evaluator.to_torch_dtype(ctx.op.dtype))

    x = ctx.args[0].tensor
    held = cast(x.tensor) if x.is_global else tuple((p, cast(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(Cast)
def cast_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)


__all__ = ["Cast"]
