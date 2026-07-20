"""HIR value-form Clamp Op: ``Clamp(x, min_val, max_val)``.

Element-wise clip: ``y = min(max(x, min_val), max_val)``.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer

# Monotone non-decreasing: commutes with max/min, not sum.
_COMMUTES_WITH = frozenset({"max", "min"})


@register_op
class Clamp(Op):
    """Element-wise clamp: y = min(max(x, min_val), max_val)."""
    x = ParamDef(kind="input", pattern=Tensor)
    min_val = ParamDef(kind="attribute", annotation=float)
    max_val = ParamDef(kind="attribute", annotation=float)

@register_typeinfer(Clamp)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )

__all__ = ["Clamp"]
