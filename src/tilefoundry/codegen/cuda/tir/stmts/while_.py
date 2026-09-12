from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.stmts import While
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

from .scalar_expr import render_scalar_expr


@register_codegen(CudaTarget, Role.EMIT, While)
def _emit(node: While, ctx: CudaCodegenContext) -> None:
    ctx.emit(f"while ({render_scalar_expr(node.cond, ctx)}) {{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
