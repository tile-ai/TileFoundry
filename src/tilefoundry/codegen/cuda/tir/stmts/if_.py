"""Emit TIR conditionals with scalar C-style predicates."""

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.stmts import If

from .scalar_expr import render_scalar_expr


@register_codegen_cuda(If)
def _emit(node: If, ctx: CodegenContext) -> None:
    cond = render_scalar_expr(node.cond, ctx)
    ctx.emit(f"if ({cond}) {{")
    ctx.indent()
    ctx.emit_node(node.then_body)
    ctx.dedent()
    if getattr(node.else_body, "body", None):
        ctx.emit("} else {")
        ctx.indent()
        ctx.emit_node(node.else_body)
        ctx.dedent()
        ctx.emit("}")
    else:
        ctx.emit("}")
