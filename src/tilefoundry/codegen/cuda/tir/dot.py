"""Emitter for the ``Dot`` TIR stmt — one uniform runtime call."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.dot import Dot
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

_DOT = "tilefoundry::ops::dot"


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, Dot)
def _emit(call, ctx: CudaCodegenContext) -> None:
    if not 3 <= len(call.args) <= 4:
        raise ValueError(
            f"tir.tensor.Dot: three operands plus an optional workspace, "
            f"got {len(call.args)} arguments"
        )
    operands = ", ".join(_tensor_expr(arg, ctx) for arg in call.args)
    ctx.emit(f"{_DOT}({operands});")
