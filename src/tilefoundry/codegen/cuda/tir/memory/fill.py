"""Codegen for ``tir.memory.Fill`` — the zero-source ``elementwise``."""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory import Fill


@register_codegen_cuda(Fill)
def _emit(call, ctx: CodegenContext) -> None:
    tensor, value = call.args[0], call.args[1]
    dst_n = ctx.name_for(tensor)
    val = value.value if isinstance(value, Constant) else 0.0
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_n}, []() {{ return {val}f; }});"
    )
