"""Codegen for ``tir.memory.Fill`` — the zero-source ``elementwise``."""
from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .tensor_view import tensor_reference


@codegen.register(target.Target, codegen.Role.EMIT, memory.Fill)
def emit(call, ctx: context.Context) -> None:
    tensor, value = call.args[0], call.args[1]
    dst_n = tensor_reference(tensor, ctx)
    val = value.value if isinstance(value, expr.Constant) else 0.0
    ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, []() {{ return {context.float_literal(val)}; }});")


__all__: list[str] = []
