"""HIR ``ShapeOf`` Op: the shape of a tensor as a rank-1 tensor."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(name="shape_of", dialect='tf', category='tensor')
class ShapeOf(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(ShapeOf)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    rank_expr = expr.Constant(type=types.TensorType.from_umat_scalar(), value=len(x_ty.shape))
    return types.TensorType.from_umat_tensor((rank_expr,), expr.DType.i64)


def shape_tensor(x):
    return utils.torch.tensor(tuple(x.shape), dtype=utils.torch.int64)


@evaluator.register(ShapeOf)
def eval_shape_of(ctx):
    x = ctx.args[0].tensor
    held = shape_tensor(x.tensor) if x.is_global else tuple((p, shape_tensor(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


access.register_relation(ShapeOf)(access.readnone_relations(lambda type_list: (len(type_list[0].shape),)))


cost.register(ShapeOf)(cost.metadata_view)


__all__ = ["ShapeOf"]
