"""Emit ``CopyAsyncBulk`` through its one runtime entry."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.codegen.tir import mbarrier
from tilefoundry_nvidia.cuda.ir.tir import memory


def tensor_expr(var, ctx: context.Context) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@codegen.register(target.Target, codegen.Role.EMIT, memory.CopyAsyncBulk)
def emit(call, ctx: context.Context) -> None:
    src = tensor_expr(call.args[0], ctx)
    dst = tensor_expr(call.args[1], ctx)
    bar = f"reinterpret_cast<uint64_t *>({mbarrier.barrier_word(call.args[2], ctx)})"
    ctx.emit(f"tilefoundry::ops::copy_async_bulk({src}, {dst}, {bar});")


__all__: list[str] = []
