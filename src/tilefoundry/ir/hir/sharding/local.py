from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import ShardLayout, Split
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    readnone_relations,
    register_access_relation,
)


@register_op
class Local(Op):
    """The current device's local view of a ``ShardLayout`` tensor."""

    x = ParamDef(kind="input", pattern=is_ranked_tensor())


def _local_shape(x_ty: TensorType) -> tuple:
    """The extents one participant holds: each static Split extent divided by its mesh."""
    layout = x_ty.layout
    if not isinstance(layout, ShardLayout):
        return tuple(x_ty.shape)
    new_shape = list(x_ty.shape)
    for mesh_axis, attr in enumerate(layout.attrs):
        if isinstance(attr, Split):
            mesh_extent = flatten(layout.mesh.layout).shape[mesh_axis]
            v = static_dim_value(new_shape[attr.axis])
            if v is not None:
                new_shape[attr.axis] = v // mesh_extent
    return tuple(new_shape)


@register_typeinfer(Local)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not isinstance(x_ty.layout, ShardLayout):
        ctx.error(call, "Local() input must have ShardLayout")
    return TensorType(
        shape=_local_shape(x_ty),
        dtype=x_ty.dtype,
        layout=x_ty.layout.layout,
        storage=x_ty.storage,
    )


@register_eval(Local)
def _eval_local(ctx):
    layout = ctx.args[0].type.layout
    if isinstance(layout, ShardLayout) and any(isinstance(attr, Split) for attr in layout.attrs):
        raise EvalError(
            "Local on a Split axis is not modelled: evaluation runs one mesh participant "
            "(docs/spec/evaluator.md section 6)."
        )

    return ctx.args[0]


register_access_relation(Local)(readnone_relations(lambda types: _local_shape(types[0])))
