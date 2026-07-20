"""Effect-ful TIR Op ``tir.nn.RMSNorm``.

Fused RMS normalization. Writes element-wise normalised output into
``dst`` by reducing ``src`` over the last axis, applying rsqrt + eps,
and multiplying by ``weight``. Wrapped by ``Evaluate(RMSNorm, ...)``
in Stmt position.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op(name="rms_norm")
class RMSNorm(Op):
    """RMS normalization writing into ``dst``."""
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    eps = ParamDef(kind="attribute", annotation=float)

@register_typeinfer(RMSNorm)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(RMSNorm)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src_ty = ctx.type_of(call.args[0])
    dst_ty = ctx.type_of(call.args[1])
    w_ty = ctx.type_of(call.args[2])

    if src_ty.shape != dst_ty.shape:
        ctx.error(
            call,
            f"tir.nn.RMSNorm shape mismatch: src {src_ty.shape} vs dst {dst_ty.shape}",
        )
    if src_ty.dtype != dst_ty.dtype:
        ctx.error(
            call,
            f"tir.nn.RMSNorm dtype mismatch: src {src_ty.dtype} vs dst {dst_ty.dtype}",
        )
    if src_ty.shape[-1] != w_ty.shape[0]:
        ctx.error(
            call,
            f"tir.nn.RMSNorm weight dim mismatch: src last dim {src_ty.shape[-1]} vs weight {w_ty.shape[0]}",
        )
    if len(w_ty.shape) != 1:
        ctx.error(
            call,
            f"tir.nn.RMSNorm weight must be 1-D, got shape {w_ty.shape}",
        )
