from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.stmts import While

from .scalar_expr import render_scalar_expr


@register_codegen_cuda(While)
def _emit(node: While, ctx: CodegenContext) -> None:
    ctx.emit(f"while ({render_scalar_expr(node.cond, ctx)}) {{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
