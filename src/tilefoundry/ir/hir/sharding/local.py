from __future__ import annotations

from dataclasses import replace

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard_layout import ShardLayout, Split, local_layout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    readnone_relations,
    register_access_relation,
)


@register_op
class Local(Op):
    """The current device's local view of a ``ShardLayout`` tensor."""

    x = ParamDef(kind="input", pattern=is_ranked_tensor())


def _local_layout(x_ty: TensorType):
    return local_layout(x_ty.layout, x_ty.shape, (0,) * len(x_ty.layout.mesh.topologies))


@register_typeinfer(Local)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not isinstance(x_ty.layout, ShardLayout):
        ctx.error(call, "Local() input must have ShardLayout")
    try:
        layout = _local_layout(x_ty)
    except ValueError as error:
        ctx.error(call, str(error))
    return replace(x_ty, shape=layout.shape, layout=layout)


@register_eval(Local)
def _eval_local(ctx):
    layout = ctx.args[0].type.layout
    if isinstance(layout, ShardLayout) and any(isinstance(attr, Split) for attr in layout.attrs):
        raise EvalError(
            "Local on a Split axis is not modelled: evaluation runs one mesh participant "
            "(docs/spec/evaluator.md section 6)."
        )

    return ctx.args[0]


register_access_relation(Local)(readnone_relations(lambda types: _local_layout(types[0]).shape))
