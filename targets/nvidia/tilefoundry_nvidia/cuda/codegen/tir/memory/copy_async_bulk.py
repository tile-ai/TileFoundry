"""Emit ``CopyAsyncBulk`` through its one runtime entry."""

from __future__ import annotations

from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.codegen.tir.mbarrier import barrier_word
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_bulk import CopyAsyncBulk


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, CopyAsyncBulk)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({barrier_word(call.args[2], ctx)})"
    ctx.emit(f"tilefoundry::ops::copy_async_bulk({src}, {dst}, {bar});")
