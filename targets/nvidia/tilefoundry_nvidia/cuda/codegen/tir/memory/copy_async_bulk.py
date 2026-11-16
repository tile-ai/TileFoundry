"""Emit ``CopyAsyncBulk`` through its one runtime entry."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.codegen.tir import mbarrier
from tilefoundry_nvidia.cuda.ir.tir import memory

from .tensor_view import emit_paired


@codegen.register(target.Target, codegen.Role.EMIT, memory.CopyAsyncBulk)
def emit(call, ctx: context.Context) -> None:
    bar = f"reinterpret_cast<uint64_t *>({mbarrier.barrier_word(call.args[2], ctx)})"
    emit_paired("tilefoundry::ops::copy_async_bulk", call.args[:2], ctx, (bar,))


__all__: list[str] = []
