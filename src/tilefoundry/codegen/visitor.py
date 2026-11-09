"""Visitor that dispatches code-generation handlers."""

from __future__ import annotations

from collections.abc import Callable

from tilefoundry import dispatch
from tilefoundry.ir import core, expr

from .registry import Role


class Visitor:
    """Dispatch a node to the handler that writes it, for a caller holding a registry.

    Not a subclass of StmtVisitor/ExprVisitor: a Stmt is reached by its own
    class and an Op through the ``Call`` carrying it, so the two arrive by
    different entries. *target* is the Target class being written, which the
    registry key carries.
    """

    def __init__(
        self,
        ctx,
        registry: dispatch.Registry,
        *,
        target: type,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.target = target
        self.registry = registry

    def lookup(self, cls: type) -> Callable:
        key = (self.target, Role.EMIT, cls)
        fn = self.registry.get(key, None)
        if fn is None:
            raise RuntimeError(f"{self.registry.name}: nothing registered for {dispatch.spelled(key)}")
        return fn

    def emit_stmt(self, stmt: core.Stmt) -> None:
        self.lookup(type(stmt))(stmt, self.ctx)

    def emit_expr(self, node: expr.Expr) -> None:
        if not isinstance(node, expr.Call):
            raise RuntimeError(
                f"codegen.Visitor.emit_expr: leaf Expr {type(node).__name__} "
                "has no emission of its own; a handler reaches it through the context."
            )
        self.lookup(type(node.target))(node, self.ctx)


__all__ = ["Visitor"]
