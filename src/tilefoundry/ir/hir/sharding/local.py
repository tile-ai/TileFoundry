"""HIR ``Local`` Op: the current device's view of a sharded tensor."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect='tf', category='sharding')
class Local(expr.Op):
    """The current device's local view of a ``ShardLayout`` tensor."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(Local)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not isinstance(x_ty.layout, types.ShardLayout):
        ctx.error(call, "Local() input must have ShardLayout")
    try:
        return core.local_type(x_ty)
    except ValueError as error:
        ctx.error(call, str(error))


@evaluator.register(Local)
def eval_local(ctx):
    """Each placement's own part: a local view unless every placement holds the whole."""
    x = ctx.args[0].tensor
    shard = types.shard_layout_of(x.type.layout)
    if not x.is_global or not any(
        isinstance(attr, (types.Split, types.Partial)) for attr in shard.attrs
    ):
        return evaluator.TensorValue(core.ShardTensor(ctx.result_type, x.tensor))
    pieces = tuple((placement, x.to_local(placement)) for placement in types.Placement.from_mesh(shard.mesh))
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, pieces))


access.register_relation(Local)(access.readnone_relations(lambda type_list: core.local_type(type_list[0]).shape))


cost.register(Local)(cost.metadata_view)


__all__ = ["Local"]
