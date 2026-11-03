from __future__ import annotations

from tilefoundry.core import ShardTensor, local_type
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.placement import units
from tilefoundry.ir.types.shard_layout import Partial, ShardLayout, Split, shard_layout_of
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
    """Each unit's own part: scattered over the mesh when an axis is Split."""
    x = ctx.args[0].tensor
    shard = shard_layout_of(x.type.layout)
    if x.packed and any(isinstance(attr, Partial) for attr in shard.attrs):
        raise EvalError(
            "Local of a Partial value: the held value is already reduced, so the "
            "part each unit holds is gone"
        )
    if x.packed and not any(isinstance(attr, Split) for attr in shard.attrs):
        return TensorValue(ShardTensor(ctx.result_type, x.tensor))
    pieces = x.scatter(tuple(placement for _, placement in units(shard.mesh))).tensor
    return TensorValue(ShardTensor(ctx.result_type, pieces))


register_access_relation(Local)(readnone_relations(lambda types: local_type(types[0]).shape))
