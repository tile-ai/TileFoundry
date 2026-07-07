"""Emitters for the ``cp.async`` TIR ops.

``CopyAsync`` forwards to ``tilefoundry::ops::copy_async``; ``CpAsyncCommit`` /
``CpAsyncWait`` emit the group-fence PTX directly.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncCommit, CpAsyncWait


def _tensor_expr(var, ctx: CodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen_cuda(CopyAsync)
def _emit_copy_async(call, ctx: CodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    ctx.emit(f"tilefoundry::ops::copy_async({src}, {dst});")


@register_codegen_cuda(CpAsyncCommit)
def _emit_commit(call, ctx: CodegenContext) -> None:
    ctx.emit('asm volatile("cp.async.commit_group;\\n" ::: "memory");')


@register_codegen_cuda(CpAsyncWait)
def _emit_wait(call, ctx: CodegenContext) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("cp.async.wait_group %0;\\n" :: "n"({n}) : "memory");')
