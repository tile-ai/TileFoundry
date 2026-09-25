"""Emit the SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.cuda.memory.ldmatrix import LdMatrix
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, LdMatrix)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    ctx.emit(f"tilefoundry::ops::ldmatrix({src}, {dst});")
