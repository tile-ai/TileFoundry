"""Codegen for ``tir.memory.Fill`` — the zero-source ``elementwise``."""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory import Fill
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, Fill)
def _emit(call, ctx: CudaCodegenContext) -> None:
    tensor, value = call.args[0], call.args[1]
    base = ctx.name_for(tensor)
    dst_n = f"{base}_tensor" if ctx.is_kernel_param(tensor) else base
    val = value.value if isinstance(value, Constant) else 0.0
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_n}, []() {{ return {val}f; }});"
    )
