"""Codegen for TIR Clamp — the pointwise entry with ``clamp_op`` as its ``fn``."""

from __future__ import annotations

from tilefoundry.ir.tir.clamp import Clamp
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext


@register_codegen(CudaTarget, Role.EMIT, Clamp)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src, dst = call.args
    op = call.target
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_n}, tilefoundry::primitive::clamp_op{{"
        f"{float(op.min_val)}f, {float(op.max_val)}f}}, {src_n});"
    )
