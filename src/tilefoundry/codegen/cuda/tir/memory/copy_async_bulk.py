"""Emit ``CopyAsyncBulk`` through its one runtime entry."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.mbarrier import barrier_word
from tilefoundry.ir.tir.cuda.memory.copy_async_bulk import CopyAsyncBulk
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, CopyAsyncBulk)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({barrier_word(call.args[2], ctx)})"
    ctx.emit(f"tilefoundry::ops::copy_async_bulk({src}, {dst}, {bar});")
