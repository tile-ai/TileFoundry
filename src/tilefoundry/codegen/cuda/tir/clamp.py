"""Codegen for TIR Clamp — the pointwise entry with ``clamp_op`` as its ``fn``."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.clamp import Clamp


@register_codegen_cuda(Clamp)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args
    op = call.target
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_n}, tilefoundry::ops::clamp_op{{"
        f"{float(op.min_val)}f, {float(op.max_val)}f}}, {src_n});"
    )
