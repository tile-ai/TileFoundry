"""Emitter for the ``Dot`` TIR stmt — one uniform runtime call."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import tir
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

DOT = "tilefoundry::ops::dot"


def tensor_expr(var, ctx: context.Context) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@codegen.register(target.Target, codegen.Role.EMIT, tir.dot.Dot)
def emit(call, ctx: context.Context) -> None:
    if not 3 <= len(call.args) <= 4:
        raise ValueError(
            f"tir.tensor.Dot: three operands plus an optional workspace, "
            f"got {len(call.args)} arguments"
        )
    operands = ", ".join(tensor_expr(arg, ctx) for arg in call.args)
    ctx.emit(f"{DOT}({operands});")


__all__: list[str] = []
