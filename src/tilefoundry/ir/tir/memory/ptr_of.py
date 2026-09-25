"""TIR view Expr Op: `tir.view.PtrOf`.

Takes a tensor Expr and returns its typed physical-memory pointer descriptor.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import Tensor
from tilefoundry.ir.types import PointerType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="ptr_of")
class PtrOf(Op):
    """Take the device address of a tensor for downstream view ops (value form)."""

    tensor = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(PtrOf)
def _(call: "Call", ctx: "TypeInferContext") -> PointerType:
    tensor = ctx.type_of(call.args[0])
    return PointerType(tensor.dtype, tensor.storage)
