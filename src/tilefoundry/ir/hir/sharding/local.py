from __future__ import annotations

from tilefoundry.core import ShardTensor, local_type
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import Placement, TensorType
from tilefoundry.ir.types.shard_layout import Broadcast, ShardLayout, shard_layout_of
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    readnone_relations,
    register_access_relation,
)


@register_op(dialect='tf', category='sharding')
class Local(Op):
    """The current device's local view of a ``ShardLayout`` tensor."""

    x = ParamDef(kind="input", pattern=is_ranked_tensor())


@register_typeinfer(Local)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not isinstance(x_ty.layout, ShardLayout):
        ctx.error(call, "Local() input must have ShardLayout")
    try:
        return local_type(x_ty)
    except ValueError as error:
        ctx.error(call, str(error))


@register_eval(Local)
def _eval_local(ctx):
    """Each placement's own part: a local view unless every placement holds the whole."""
    x = ctx.args[0].tensor
    shard = shard_layout_of(x.type.layout)
    if not x.is_global or all(isinstance(attr, Broadcast) for attr in shard.attrs):
        return TensorValue(ShardTensor(ctx.result_type, x.tensor))
    pieces = tuple((placement, x.to_local(placement)) for placement in Placement.each(shard.mesh))
    return TensorValue(ShardTensor(ctx.result_type, pieces))


register_access_relation(Local)(readnone_relations(lambda types: local_type(types[0]).shape))
