"""Emit the SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import memory


def tensor_expr(var, ctx: context.Context) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@codegen.register(target.Target, codegen.Role.EMIT, memory.LdMatrix)
def emit(call, ctx: context.Context) -> None:
    src = tensor_expr(call.args[0], ctx)
    dst = tensor_expr(call.args[1], ctx)
    ctx.emit(f"tilefoundry::ops::ldmatrix({src}, {dst});")


__all__: list[str] = []
