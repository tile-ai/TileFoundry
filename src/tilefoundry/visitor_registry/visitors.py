"""Verification, code generation, and cost-evaluation visitors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tilefoundry.ir.core import Op, VerifyError
from tilefoundry.ir.core.expr import Call, Expr
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.visitor import ExprWalker, StmtVisitor

from .contexts import Cost, CostContext, VerifyContext
from .instruction import ScheduleRefusal, check_call
from .registries import (
    DispatchRegistry,
    Role,
    cost_evaluator_registry,
    spelled,
    verify_stmt_registry,
)


class VerifyVisitor(StmtVisitor[None]):
    """Dispatch verify_stmt per Stmt subclass.

    Unregistered Stmt subclasses (typically control-flow: For/While/If/
    Assign/MeshScope) fall through to the StmtVisitor default traversal,
    which recurses into children without raising. That is intentional —
    control-flow stmts whose semantics are fully captured by structure need
    no custom verify.
    """

    def __init__(
        self,
        ctx: VerifyContext,
        registry: DispatchRegistry = verify_stmt_registry,
    ) -> None:
        self.ctx = ctx
        self.registry = registry

    def generic_visit(self, stmt: Stmt) -> None:
        if isinstance(stmt, Evaluate):
            op = stmt.callable
            if isinstance(op, Op):
                try:
                    check_call(op, stmt.args, self.ctx.current_mesh, frame=self.ctx)
                except ScheduleRefusal as error:
                    raise VerifyError(str(error)) from error
            super().generic_visit(stmt)
            return
        fn = self.registry.lookup(type(stmt))
        if fn is not None:
            fn(stmt, self.ctx)
        super().generic_visit(stmt)

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        fn = self.registry.lookup(MeshScope)
        if fn is not None:
            fn(stmt, self.ctx)
        outer = self.ctx
        self.ctx = replace(
            outer,
            current_mesh=make_mesh(outer.current_mesh, stmt.mesh) if outer.current_mesh else stmt.mesh,
            mesh_scope=(*outer.mesh_scope, stmt.mesh),
        )
        try:
            self.visit(stmt.body)
        finally:
            self.ctx = outer



class CodegenVisitor:
    """Dispatch a node to the handler that writes it, for a caller holding a registry.

    Not a subclass of StmtVisitor/ExprVisitor: a Stmt is reached by its own
    class and an Op through the ``Call`` carrying it, so the two arrive by
    different entries. *target* is the Target class being written, which the
    registry key carries.
    """

    def __init__(
        self,
        ctx,
        registry: DispatchRegistry,
        *,
        target: type,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.target = target
        self.registry = registry

    def _handler(self, cls: type) -> Callable:
        key = (self.target, Role.EMIT, cls)
        fn = self.registry.lookup(key)
        if fn is None:
            raise RuntimeError(f"{self.registry.name}: nothing registered for {spelled(key)}")
        return fn

    def emit_stmt(self, stmt: Stmt) -> None:
        self._handler(type(stmt))(stmt, self.ctx)

    def emit_expr(self, expr: Expr) -> None:
        if not isinstance(expr, Call):
            raise RuntimeError(
                f"CodegenVisitor.emit_expr: leaf Expr {type(expr).__name__} "
                "has no emission of its own; a handler reaches it through the context."
            )
        self._handler(type(expr.target))(expr, self.ctx)


class CostEvaluator(ExprWalker[Cost]):
    """Dispatch the registered recursive-local Cost Evaluator per Op class.

    A missing evaluator fails closed — it is a construction error, not a
    zero-Cost default.
    """

    def __init__(
        self,
        registry: DispatchRegistry = cost_evaluator_registry,
    ) -> None:
        super().__init__()
        self.registry = registry

    def visit_Call(self, call: Call, ctx: CostContext) -> Cost:
        fn = self.registry.lookup(type(call.target))
        if fn is None:
            ctx.error(call, f"no cost evaluator registered for {type(call.target).__name__}")
        return fn(call, ctx)


__all__ = ["VerifyVisitor", "CodegenVisitor", "CostEvaluator"]
