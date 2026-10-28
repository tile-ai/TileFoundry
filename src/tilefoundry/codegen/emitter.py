"""C++ source emitters shared by target code generators."""

from __future__ import annotations

from tilefoundry.codegen.scalar_expr import render_scalar_expr
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.tir.abort import Abort
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.target.base import target_summary


class CppEmitter(StmtVisitor[None]):
    """Write C++ from TIR; the context carries state, this class traverses."""

    def __init__(self, *, context) -> None:
        self.context = context

    def visit(self, node):
        handler = self.context.stmt_handler(type(node))
        return handler(node, self) if handler is not None else super().visit(node)

    def emit(self, line: str) -> None:
        self.context.emit(line)

    def indent(self) -> None:
        self.context.indent()

    def dedent(self) -> None:
        self.context.dedent()

    def statement(self, text: str) -> None:
        self.emit(f"{text};")

    def generic_visit(self, node) -> None:
        raise RuntimeError(f"no codegen handler for {type(node).__name__} on {target_summary(self.context.target)}")

    def visit_Sequential(self, node: Sequential) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: If) -> None:
        cond = self.render_scalar(node.cond)
        self.emit(f"if ({cond}) {{")
        self.indent()
        self.visit(node.then_body)
        self.dedent()
        if getattr(node.else_body, "body", None):
            self.emit("} else {")
            self.indent()
            self.visit(node.else_body)
            self.dedent()
        self.emit("}")

    def visit_For(self, node: For) -> None:
        name = self.context.name_for(node.induction_var)
        start = self.render_scalar(node.start)
        stop = self.render_scalar(node.stop)
        step = self.render_scalar(node.step)
        self.emit(f"for (int {name} = {start}; {name} < {stop}; {name} += {step}) {{")
        self.indent()
        self.visit(node.body)
        self.dedent()
        self.emit("}")

    def visit_While(self, node: While) -> None:
        self.emit(f"while ({self.render_scalar(node.cond)}) {{")
        self.indent()
        self.visit(node.body)
        self.dedent()
        self.emit("}")

    def visit_Return(self, node: Return) -> None:
        self.statement("return")

    def visit_Abort(self, node: Abort) -> None:
        self.statement("__trap()")

    def visit_LetStmt(self, node: LetStmt) -> None:
        """Emit a call result or let the target materialize a tuple binding."""
        if isinstance(node.value, Tuple):
            self.context.register_tuple(node.var, node.value)
            if (handler := self.context.stmt_handler(Tuple)) is not None:
                handler(node, self.context)
            return
        if not isinstance(node.value, Call):
            raise RuntimeError(
                f"LetStmt.value must be a Call (TIR-owned Expr Op), got {type(node.value).__name__}"
            )
        self.emit_let_op(node.value, node)

    def visit_Evaluate(self, node: Evaluate) -> None:
        op = node.callable
        if isinstance(op, (PrimFunction, ModuleList, Call)):
            raise RuntimeError(f"no codegen handler for Op {type(op).__name__} on {target_summary(self.context.target)}")
        self.emit_evaluate_op(Call(type=UnitType(), target=op, args=node.args))

    def emit_evaluate_op(self, call: Call) -> None:
        handler = self.context.handler_for(type(call.target))
        handler(call, self.context)

    def emit_let_op(self, call: Call, stmt: LetStmt) -> None:
        handler = self.context.handler_for(type(call.target))
        handler(stmt, self.context)

    def render_scalar(self, expr) -> str:
        return render_scalar_expr(expr, self.context)


__all__ = ["CppEmitter"]
