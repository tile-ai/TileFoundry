"""Effect-ful TIR Op ``tir.nn.ReLU``.

Tensor-level pointwise ReLU — writes element-wise ``max(src, 0)``
into ``dst`` (in-place memory write). Wrapped by
``Evaluate(ReLU, ...)`` in Stmt position.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.types import UnitType


@register_op
class ReLU(Op):
    """Tensor-level pointwise ReLU writing into ``dst``.

    Spec: tir.md §3.2
    """
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)

@register_typeinfer(ReLU)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(ReLU)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src_ty = ctx.type_of(call.args[0])
    dst_ty = ctx.type_of(call.args[1])
    if src_ty.shape != dst_ty.shape:
        ctx.error(call, f"nn.ReLU shape mismatch: {src_ty.shape} vs {dst_ty.shape}")
    if src_ty.dtype != dst_ty.dtype:
        ctx.error(call, f"nn.ReLU dtype mismatch: {src_ty.dtype} vs {dst_ty.dtype}")
