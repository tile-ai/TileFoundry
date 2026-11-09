"""Statement visitor that dispatches verification rules."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry import dispatch, services
from tilefoundry.ir import core, expr, types

from .context import Context
from .registry import registry


class Visitor(core.StmtVisitor[None]):
    """Dispatch verify_stmt per Stmt subclass.

    Unregistered Stmt subclasses (typically control-flow: For/While/If/
    Assign/MeshScope) fall through to the StmtVisitor default traversal,
    which recurses into children without raising. That is intentional —
    control-flow stmts whose semantics are fully captured by structure need
    no custom verify.
    """

    def __init__(
        self,
        ctx: Context,
        registry: dispatch.Registry = registry,
    ) -> None:
        self.ctx = ctx
        self.registry = registry

    def generic_visit(self, stmt: core.Stmt) -> None:
        if isinstance(stmt, core.Evaluate):
            op = stmt.callable
            if isinstance(op, expr.Op):
                services.verify_call(op, stmt.args, self.ctx.current_mesh, frame=self.ctx)
            super().generic_visit(stmt)
            return
        fn = self.registry.get(type(stmt), None)
        if fn is not None:
            fn(stmt, self.ctx)
        super().generic_visit(stmt)

    def visit_MeshScope(self, stmt: core.MeshScope) -> None:
        fn = self.registry.get(core.MeshScope, None)
        if fn is not None:
            fn(stmt, self.ctx)
        outer = self.ctx
        self.ctx = replace(
            outer,
            current_mesh=types.Mesh.from_meshes(outer.current_mesh, stmt.mesh)
            if outer.current_mesh
            else stmt.mesh,
            mesh_scope=(*outer.mesh_scope, stmt.mesh),
        )
        try:
            self.visit(stmt.body)
        finally:
            self.ctx = outer


__all__ = ["Visitor"]
