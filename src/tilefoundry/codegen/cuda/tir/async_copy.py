"""Emitters for the ``cp.async`` TIR ops.

``CopyAsync`` forwards to ``tilefoundry::ops::copy_async``; ``CpAsyncCommit`` /
``CpAsyncWait`` emit the group-fence PTX directly.
"""

from __future__ import annotations

import math

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.async_copy import CopyAsync, CpAsyncCommit, CpAsyncWait, indexed_width
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _tensor_expr(var, ctx: CudaCodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen(CudaTarget, Role.EMIT, CopyAsync)
def _emit_copy_async(call, ctx: CudaCodegenContext) -> None:
    src = _tensor_expr(call.args[0], ctx)
    dst = _tensor_expr(call.args[1], ctx)
    if len(call.args) == 3:
        index = _tensor_expr(call.args[2], ctx)
        width = indexed_width(call.args[0].type, call.args[1].type)
        mesh_type = next(reversed(ctx._mesh_aliases.values()))[0]
        fill = call.target.fill
        suffix = ""
        if fill is not None:
            if math.isinf(fill):
                literal = "__int_as_float(0xff800000)" if fill < 0 else "__int_as_float(0x7f800000)"
            elif math.isnan(fill):
                literal = "__int_as_float(0x7fc00000)"
            else:
                literal = repr(float(fill)) + "f"
            suffix = f", {literal}"
        ctx.emit(
            f"tilefoundry::ops::copy_async<{mesh_type}, {width}>({src}, {dst}, {index}{suffix});"
        )
        return
    ctx.emit(f"tilefoundry::ops::copy_async({src}, {dst});")


@register_codegen(CudaTarget, Role.EMIT, CpAsyncCommit)
def _emit_commit(call, ctx: CudaCodegenContext) -> None:
    ctx.emit('asm volatile("cp.async.commit_group;\\n" ::: "memory");')


@register_codegen(CudaTarget, Role.EMIT, CpAsyncWait)
def _emit_wait(call, ctx: CudaCodegenContext) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("cp.async.wait_group %0;\\n" :: "n"({n}) : "memory");')
