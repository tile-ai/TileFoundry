"""Emitters for the ``cp.async`` TIR ops.

``CopyAsync`` forwards to ``tilefoundry::ops::copy_async``; ``CpAsyncCommit`` /
``CpAsyncWait`` emit the group-fence PTX directly.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncCommit, CpAsyncWait
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, CopyAsync)
def _emit_copy_async(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    ctx.emit(f"tilefoundry::ops::copy_async({src}, {dst});")


@register_codegen(CudaTarget, Role.EMIT, CpAsyncCommit)
def _emit_commit(call, ctx: CudaCodegenContext) -> None:
    ctx.emit('asm volatile("cp.async.commit_group;\\n" ::: "memory");')


@register_codegen(CudaTarget, Role.EMIT, CpAsyncWait)
def _emit_wait(call, ctx: CudaCodegenContext) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("cp.async.wait_group %0;\\n" :: "n"({n}) : "memory");')
