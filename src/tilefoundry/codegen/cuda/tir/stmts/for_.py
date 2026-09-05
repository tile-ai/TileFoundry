"""Emitter for ``tir.For`` — C-style for loop."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.stmts import For

from .scalar_expr import render_scalar_expr


@register_codegen_cuda(For)
def _emit(node: For, ctx: CodegenContext) -> None:
    iv_name = ctx.name_for(node.induction_var)
    start = render_scalar_expr(node.start, ctx)
    stop = render_scalar_expr(node.stop, ctx)
    step = render_scalar_expr(node.step, ctx)

    ctx.emit(f"for (int {iv_name} = {start}; {iv_name} < {stop}; {iv_name} += {step}) {{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
