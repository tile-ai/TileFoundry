"""TIR view Expr Op: `tir.view.PtrOf`.

Takes a tensor Expr and returns its typed physical-memory pointer descriptor.
"""

from __future__ import annotations

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(name="ptr_of", dialect='T', category='memory')
class PtrOf(expr.Op):
    """Take the device address of a tensor for downstream view ops (value form)."""

    tensor = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(PtrOf)
def _(call: "Call", ctx: "TypeInferContext") -> types.PointerType:
    tensor = ctx.type_of(call.args[0])
    return types.PointerType(tensor.dtype, tensor.storage)


__all__ = ["PtrOf"]
