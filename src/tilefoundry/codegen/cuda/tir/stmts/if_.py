"""Emit TIR conditionals with scalar C-style predicates.

Supported predicates are integer or boolean constants, variables, and
value-form comparison or logical binary calls. Tensor-form binary operations
are emitted elsewhere. Unsupported expressions raise ``NotImplementedError``
instead of producing an incorrect predicate.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.tir.stmts import If

_SCALAR_BINARY_OP: dict[BinaryKind, str] = {
    BinaryKind.EQ: "==",
    BinaryKind.NE: "!=",
    BinaryKind.LT: "<",
    BinaryKind.LE: "<=",
    BinaryKind.GT: ">",
    BinaryKind.GE: ">=",
    BinaryKind.AND: "&&",
}


def render_scalar_predicate(expr, ctx: CodegenContext) -> str:
    """Render a scalar boolean / integer expression as a C source string.

    Intended for ``tir.If.cond`` and any other scalar predicate site.
    Walks only the small Expr subset listed in the module docstring.
    """
    if isinstance(expr, Constant):
        value = expr.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        raise NotImplementedError(
            f"render_scalar_predicate: Constant value of type "
            f"{type(value).__name__!r} is not supported "
            f"(only int / bool)."
        )
    if isinstance(expr, Var):
        return ctx.name_for(expr)
    if isinstance(expr, Call):
        op = expr.target
        kind = getattr(op, "kind", None)
        if not isinstance(kind, BinaryKind) or kind not in _SCALAR_BINARY_OP:
            raise NotImplementedError(
                f"render_scalar_predicate: Call target {type(op).__name__} "
                f"with kind {kind!r} is not a supported scalar binary. "
                f"Supported kinds: "
                f"{sorted(k.name for k in _SCALAR_BINARY_OP)}."
            )
        if len(expr.args) != 2:
            raise ValueError(
                f"render_scalar_predicate: scalar Binary expects 2 args, "
                f"got {len(expr.args)}"
            )
        lhs = render_scalar_predicate(expr.args[0], ctx)
        rhs = render_scalar_predicate(expr.args[1], ctx)
        return f"({lhs}) {_SCALAR_BINARY_OP[kind]} ({rhs})"
    raise NotImplementedError(
        f"render_scalar_predicate: Expr type {type(expr).__name__!r} is "
        f"not supported."
    )


@register_codegen_cuda(If)
def _emit(node: If, ctx: CodegenContext) -> None:
    cond = render_scalar_predicate(node.cond, ctx)
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


__all__ = ["render_scalar_predicate"]
