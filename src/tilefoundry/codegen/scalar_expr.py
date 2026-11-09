"""Shared scalar Expr to C renderer for TIR statement emitters."""

from types import MappingProxyType

from tilefoundry.ir import core, expr

from .signature import tensor_signature_of

SCALAR_BINARY_OP = MappingProxyType({
    expr.BinaryKind.EQ: "==",
    expr.BinaryKind.NE: "!=",
    expr.BinaryKind.LT: "<",
    expr.BinaryKind.LE: "<=",
    expr.BinaryKind.GT: ">",
    expr.BinaryKind.GE: ">=",
    expr.BinaryKind.AND: "&&",
})


class PredicateVisitor(core.ExprVisitor[str]):
    def visit_Constant(self, node: expr.Constant, ctx: "EmitContext") -> str:
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, int):
            return str(node.value)
        raise NotImplementedError(
            f"render_scalar_expr: Constant value of type {type(node.value).__name__!r} "
            "is not supported (only int / bool)."
        )

    def visit_Var(self, node: expr.Var, ctx: "EmitContext") -> str:
        return ctx.name_for(node)

    def visit_ShapeOf(self, node: core.ShapeOf, ctx: "EmitContext") -> str:
        dim = node.param.type.shape[node.axis]
        return (
            str(dim)
            if isinstance(dim, int)
            else ctx.local_extent(tensor_signature_of(node.param), node.axis)
        )

    def visit_Call(self, node: expr.Call, ctx: "EmitContext") -> str:
        kind = getattr(node.target, "kind", None)
        if not isinstance(kind, expr.BinaryKind):
            return ctx.handler_for(type(node.target))(node, ctx)
        if kind not in SCALAR_BINARY_OP:
            raise NotImplementedError(
                f"render_scalar_expr: Call target {type(node.target).__name__} "
                f"with kind {kind!r} is not a supported scalar binary. "
                f"Supported kinds: {sorted(k.name for k in SCALAR_BINARY_OP)}."
            )
        if len(node.args) != 2:
            raise ValueError(
                f"render_scalar_expr: scalar Binary expects 2 args, got {len(node.args)}"
            )
        lhs, rhs = (self.visit(arg, ctx) for arg in node.args)
        return f"({lhs}) {SCALAR_BINARY_OP[kind]} ({rhs})"

    def default_visit(self, node, ctx: "EmitContext") -> str:
        raise NotImplementedError(
            f"render_scalar_expr: Expr type {type(node).__name__!r} is not supported."
        )


def render_scalar_expr(node, ctx: "EmitContext") -> str:
    return PredicateVisitor().visit(node, ctx)


__all__ = ["render_scalar_expr"]
