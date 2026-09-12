"""Emitter for ``TmaCopy`` — one line, whichever instruction ends up running."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.mbarrier import barrier_word
from tilefoundry.ir.tir.cuda.memory.tma import TmaCopy
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

_TMA_COPY = "tilefoundry::ops::tma_copy"


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, TmaCopy)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({barrier_word(call.args[2], ctx)})"
    ctx.emit(f"{_TMA_COPY}({src}, {dst}, {bar});")
