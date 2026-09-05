"""Shared scalar Expr to C renderer for TIR statement emitters."""
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.visitor import ExprVisitor

_SCALAR_BINARY_OP = {BinaryKind.EQ: "==", BinaryKind.NE: "!=", BinaryKind.LT: "<", BinaryKind.LE: "<=", BinaryKind.GT: ">", BinaryKind.GE: ">=", BinaryKind.AND: "&&"}

class _PredicateVisitor(ExprVisitor[str]):
    def visit_Constant(self, expr: Constant, ctx: CodegenContext) -> str:
        if isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        if isinstance(expr.value, int):
            return str(expr.value)
        raise NotImplementedError(
            f"render_scalar_expr: Constant value of type {type(expr.value).__name__!r} "
            "is not supported (only int / bool)."
        )
    def visit_Var(self, expr: Var, ctx: CodegenContext) -> str:
        return ctx.name_for(expr)
    def visit_Call(self, expr: Call, ctx: CodegenContext) -> str:
        kind = getattr(expr.target, "kind", None)
        if not isinstance(kind, BinaryKind) or kind not in _SCALAR_BINARY_OP:
            raise NotImplementedError(
                f"render_scalar_expr: Call target {type(expr.target).__name__} "
                f"with kind {kind!r} is not a supported scalar binary. "
                f"Supported kinds: {sorted(k.name for k in _SCALAR_BINARY_OP)}."
            )
        if len(expr.args) != 2:
            raise ValueError(
                f"render_scalar_expr: scalar Binary expects 2 args, got {len(expr.args)}"
            )
        lhs, rhs = (self.visit(arg, ctx) for arg in expr.args)
        return f"({lhs}) {_SCALAR_BINARY_OP[kind]} ({rhs})"
    def default_visit(self, expr, ctx: CodegenContext) -> str:
        raise NotImplementedError(
            f"render_scalar_expr: Expr type {type(expr).__name__!r} is not supported."
        )

def render_scalar_expr(expr, ctx: CodegenContext) -> str:
    return _PredicateVisitor().visit(expr, ctx)
