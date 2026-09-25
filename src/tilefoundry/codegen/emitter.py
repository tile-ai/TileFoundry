"""C++ source emitters shared by CUDA and CPU code generation."""

from __future__ import annotations

from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import (
    _is_dynamic_mesh,
    _validate_topology,
    mesh_type,
    program_topologies,
)
from tilefoundry.codegen.cuda.tir.stmts.scalar_expr import render_scalar_expr
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.tir.abort import Abort
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.types import ComposedLayout, UnitType
from tilefoundry.ir.visitor import StmtVisitor


class CppEmitter(StmtVisitor[None]):
    """Write C++ from TIR; the context carries state, this class traverses."""

    def __init__(self, *, context) -> None:
        self.context = context

    def emit(self, line: str) -> None:
        self.context.emit(line)

    def indent(self) -> None:
        self.context.indent()

    def dedent(self) -> None:
        self.context.dedent()

    def statement(self, text: str) -> None:
        self.emit(f"{text};")

    def generic_visit(self, node) -> None:
        raise RuntimeError(f"no emitter for {type(node).__name__}")

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
        """Emit a stored call result, or pass through a structural tuple binding.

        A Tuple has no storage of its own. Its consumers lower the elements
        directly, so its authored name is absent from generated C++ and each
        TupleGetItem use may materialize its own target-side aggregate.
        """
        if isinstance(node.value, Tuple):
            self.context.register_tuple(node.var, node.value)
            self.visit(node.body)
            return
        if not isinstance(node.value, Call):
            raise RuntimeError(
                f"LetStmt.value must be a Call (TIR-owned Expr Op), got {type(node.value).__name__}"
            )
        self.emit_let_op(node.value, node)
        self.visit(node.body)

    def visit_Evaluate(self, node: Evaluate) -> None:
        op = node.callable
        self.emit_evaluate_op(Call(type=UnitType(), target=op, args=node.args))

    def visit_MeshScope(self, node: MeshScope) -> None:
        raise NotImplementedError

    def emit_evaluate_op(self, call: Call) -> None:
        handler = self.context.handler_for(type(call.target))
        if handler is None:
            raise RuntimeError(f"no codegen handler for Op {type(call.target).__name__}")
        handler(call, self.context)

    def emit_let_op(self, call: Call, stmt: LetStmt) -> None:
        handler = self.context.handler_for(type(call.target))
        if handler is None:
            raise RuntimeError(f"no codegen handler for Op {type(call.target).__name__}")
        handler(stmt, self.context)

    def render_scalar(self, expr) -> str:
        return render_scalar_expr(expr, self.context)


class CudaEmitter(CppEmitter):
    """CUDA C++ emitter, including mesh scope lowering."""

    def visit_MeshScope(self, node: MeshScope) -> None:
        ctx = self.context
        if ctx.target is None:
            raise RuntimeError("CUDA MeshScope emission requires its Target")
        _validate_topology(node.mesh, ctx.target)
        name = ctx.name_for(node.binding)
        ctx.emit(f"// mesh scope: {program_topologies(node.mesh)[0].name}")

        is_slice = isinstance(node.mesh.layout, ComposedLayout)
        ctx.emit("{")
        ctx.indent()
        outer_aliases = ctx._mesh_aliases
        ctx._mesh_aliases = dict(outer_aliases)
        try:
            if not _is_dynamic_mesh(node.mesh):
                alias = f"{name}_mesh_t"
                mesh_type_str = mesh_type(node.mesh)
                ctx._mesh_aliases[id(node.mesh)] = (alias, mesh_type_str)
                ctx.emit(f"using {alias} = {mesh_type_str};")
                ctx.emit(f"constexpr {alias} {name}_mesh{{}};")
            if is_slice:
                ctx.emit(f"if (tilefoundry::contains({name}_mesh, tilefoundry::program_ids())) {{")
                ctx.indent()
            self.visit(node.body)
            if is_slice:
                ctx.dedent()
                ctx.emit("}")
        finally:
            ctx._mesh_aliases = outer_aliases
        ctx.dedent()
        ctx.emit("}")


__all__ = ["CppEmitter", "CudaEmitter"]
