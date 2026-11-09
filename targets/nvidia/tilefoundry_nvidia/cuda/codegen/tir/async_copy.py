"""Emitters for the ``cp.async`` TIR ops.

``CopyAsync`` forwards to ``tilefoundry::ops::copy_async``; ``CpAsyncCommit`` /
``CpAsyncWait`` emit the group-fence PTX directly.
"""

from __future__ import annotations

import math

from tilefoundry import codegen
from tilefoundry.ir import tir
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context


def tensor_expr(var, ctx: context.Context) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@codegen.register(target.Target, codegen.Role.EMIT, tir.async_copy.CopyAsync)
def emit_copy_async(call, ctx: context.Context) -> None:
    src = tensor_expr(call.args[0], ctx)
    dst = tensor_expr(call.args[1], ctx)
    if tir.async_copy.is_indexed_copy(call.args):
        index = tensor_expr(call.args[2], ctx)
        width = tir.async_copy.indexed_width(call.args[0].type, call.args[1].type)
        mesh_type = next(reversed(ctx.mesh_aliases.values()))[0]
        fill = call.target.fill
        suffix = ""
        if fill is not None:
            if math.isinf(fill):
                literal = "__int_as_float(0xff800000)" if fill < 0 else "__int_as_float(0x7f800000)"
            elif math.isnan(fill):
                literal = "__int_as_float(0x7fc00000)"
            else:
                literal = context.float_literal(fill)
            suffix = f", {literal}"
        ctx.emit(
            f"tilefoundry::ops::copy_async<{mesh_type}, {width}>({src}, {dst}, {index}{suffix});"
        )
        return
    ctx.emit(f"tilefoundry::ops::copy_async({src}, {dst});")


@codegen.register(target.Target, codegen.Role.EMIT, tir.async_copy.CpAsyncCommit)
def emit_commit(call, ctx: context.Context) -> None:
    ctx.emit('asm volatile("cp.async.commit_group;\\n" ::: "memory");')


@codegen.register(target.Target, codegen.Role.EMIT, tir.async_copy.CpAsyncWait)
def emit_wait(call, ctx: context.Context) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("cp.async.wait_group %0;\\n" :: "n"({n}) : "memory");')


__all__: list[str] = []
