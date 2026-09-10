"""Emitter for ``TmaCopy`` — one line, whichever instruction ends up running."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.mbarrier import barrier_word
from tilefoundry.ir.tir.cuda.memory.tma import TmaCopy

_TMA_COPY = "tilefoundry::ops::tma_copy"


def _tensor_expr(var, ctx: CodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen_cuda(TmaCopy)
def _emit(call, ctx: CodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({barrier_word(call.args[2], ctx)})"
    ctx.emit(f"{_TMA_COPY}({src}, {dst}, {bar});")
