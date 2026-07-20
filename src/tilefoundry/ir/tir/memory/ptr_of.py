"""TIR view Expr Op: `tir.view.PtrOf`.

Takes an Expr of tensor/scalar type, returns a raw
pointer descriptor. Placeholder: typeinfer returns the input type until a
dedicated PointerType lands.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="ptr_of")
class PtrOf(Op):
    """Take the device address of a tensor for downstream view ops (value form)."""
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(PtrOf)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return ctx.type_of(call.args[0])
