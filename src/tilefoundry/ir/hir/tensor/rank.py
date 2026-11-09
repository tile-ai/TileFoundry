"""HIR ``Rank`` Op: the rank of a tensor as a scalar."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect='tf', category='tensor')
class Rank(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(Rank)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    return types.TensorType.from_umat_scalar()


def rank_tensor(x):
    return utils.torch.tensor(x.ndim, dtype=utils.torch.int64)


@evaluator.register(Rank)
def eval_rank(ctx):
    x = ctx.args[0].tensor
    held = rank_tensor(x.tensor) if x.is_global else tuple((p, rank_tensor(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


access.register_relation(Rank)(access.readnone_relations(lambda type_list: ()))


cost.register(Rank)(cost.metadata_view)


__all__ = ["Rank"]
