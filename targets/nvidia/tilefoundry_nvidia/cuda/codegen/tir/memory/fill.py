"""Codegen for ``tir.memory.Fill`` — the zero-source ``elementwise``."""
from __future__ import annotations

from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory import Fill
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext, float_literal
from tilefoundry_nvidia.cuda.codegen.tir.memory.tensor_view import _tensor_ref


@register_codegen(CudaTarget, Role.EMIT, Fill)
def _emit(call, ctx: CudaCodegenContext) -> None:
    tensor, value = call.args[0], call.args[1]
    dst_n = _tensor_ref(tensor, ctx)
    val = value.value if isinstance(value, Constant) else 0.0
    ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, []() {{ return {float_literal(val)}; }});")
