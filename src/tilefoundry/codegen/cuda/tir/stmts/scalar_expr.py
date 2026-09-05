"""Render scalar TIR expressions for statement emitters."""
from tilefoundry.codegen.cuda.context import CodegenContext

from .if_ import _PredicateVisitor


def render_scalar_expr(expr, ctx: CodegenContext) -> str:
    return _PredicateVisitor().visit(expr, ctx)
