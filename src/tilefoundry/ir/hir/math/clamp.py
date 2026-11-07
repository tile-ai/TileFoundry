"""HIR value-form Clamp Op: ``Clamp(x, min_val, max_val)``.

Element-wise clip: ``y = min(max(x, min_val), max_val)``.
"""

from __future__ import annotations

from tilefoundry.core import ShardTensor
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.utils import reject_partials
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)

_COMMUTES_WITH = frozenset({"max", "min"})


@register_op(dialect='tf', category='math')
class Clamp(Op):
    """Element-wise clamp: y = min(max(x, min_val), max_val)."""

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    min_val = ParamDef(kind="attribute", annotation=float)
    max_val = ParamDef(kind="attribute", annotation=float)


@register_typeinfer(Clamp)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials("x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )


@register_eval(Clamp)
def _eval_clamp(ctx):
    def clamp(x):
        return torch.clamp(x, min=ctx.op.min_val, max=ctx.op.max_val)

    x = ctx.args[0].tensor
    held = clamp(x.tensor) if x.is_global else tuple((p, clamp(t)) for p, t in x.tensor)
    return TensorValue(ShardTensor(ctx.result_type, held))


__all__ = ["Clamp"]


register_access_relation(Clamp)(identity_relations)
