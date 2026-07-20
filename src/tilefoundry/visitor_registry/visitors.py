"""Derived Visitors — TypeInferVisitor / VerifyVisitor / CodegenVisitor / CostEvaluator.

`AnalysisRegistry` instance with a traversal skeleton from
tilefoundry.ir.visitor.

The `registry` is exposed as an advanced constructor param (default: the
canonical module-level registry for that analysis). Default path uses the
module-level registry directly; passing a custom one is an advanced
extension point for sandbox tests or grouped dispatch.
"""
from __future__ import annotations

from tilefoundry.ir.core.expr import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope
from tilefoundry.ir.types.tensor_type import TupleType, Type, UnitType
from tilefoundry.ir.visitor import ExprVisitor, StmtVisitor

from .contexts import Cost, CostContext, TypeInferContext, VerifyContext, _constant_type
from .registries import (
    AnalysisRegistry,
    codegen_cpu_registry,
    codegen_cuda_registry,
    cost_evaluator_registry,
    typeinfer_registry,
    verify_stmt_registry,
)


class TypeInferVisitor(ExprVisitor[Type]):
    """The one typeinfer derivation rule per ``Expr`` kind. hir.md §1.1,
    visitor-registry.md §4.

    ``TypeInferContext.type_of`` is the caller-facing cache + dispatch
    entry; it constructs one of these per lookup and delegates to
    ``visit(expr)``. There is no ``isinstance`` fallback — an ``Expr``
    subclass with no ``visit_<Kind>`` here raises via ``generic_visit``
    rather than trusting a possibly-stale ``expr.type`` field.
    """

    def __init__(self, ctx: TypeInferContext) -> None:
        self.ctx = ctx

    def visit_Var(self, var: Var) -> Type:
        return var.type

    def visit_Constant(self, c: Constant) -> Type:
        declared = c.type
        if declared is not None:
            return declared
        return _constant_type(c.value)

    def visit_Call(self, call: Call) -> Type:
        op_cls = type(call.target)
        fn = typeinfer_registry.lookup(op_cls)
        if fn is None:
            self.ctx.error(call, f"no typeinfer registered for {op_cls.__name__}")
        return fn(call, self.ctx)

    def visit_Tuple(self, tup: Tuple) -> Type:
        """Structural: the field types of the (possibly just-elaborated)
        elements, never the node's own stamped ``.type`` (hir.md §1.1)."""
        return TupleType(fields=tuple(self.ctx.type_of(e) for e in tup.elements))

    def visit_GridRegionExpr(self, grid: GridRegionExpr) -> Type:
        """Carry/body: a no-carry loop's value is its body; a carrying loop's
        value is its ``carried_args`` phi Vars' own declared type(s) — the
        same rule the parser applies when constructing the node (hir.md
        §1.2)."""
        self.ctx.type_of(grid.body)
        for y in grid.yield_values:
            self.ctx.type_of(y)
        if not grid.carried_args:
            return self.ctx.type_of(grid.body)
        if len(grid.carried_args) == 1:
            return grid.carried_args[0].type
        return TupleType(fields=tuple(p.type for p in grid.carried_args))

    def visit_ShapeOf(self, shape_of: ShapeOf) -> Type:
        """A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32)
        type at construction; it has no children to derive from."""
        return shape_of.type

    def generic_visit(self, expr: Expr) -> Type:
        self.ctx.error(expr, f"no typeinfer rule for Expr subclass {type(expr).__name__}")


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
        registry: AnalysisRegistry = verify_stmt_registry,
    ) -> None:
        # Default path: no registry argument, use the module-level
        # verify_stmt_registry. Passing an explicit registry is an advanced
        # extension point (e.g. sandbox tests, grouped dispatch); the everyday
        # verify pass never needs it.
        self.ctx = ctx
        self.registry = registry

    def generic_visit(self, stmt: Stmt) -> None:
        if isinstance(stmt, Evaluate):
            # Effect-ful Op invocation in Stmt position: dispatch verify on the
            # Op class. The handler ABI is Call-based, so feed it a Call built
            # from the Op and its args.
            op = stmt.callable
            fn = self.registry.lookup(type(op))
            if fn is not None:
                call = Call(type=UnitType(), target=op, args=stmt.args)
                fn(call, self.ctx)
            super().generic_visit(stmt)
            return
        fn = self.registry.lookup(type(stmt))
        if fn is not None:
            fn(stmt, self.ctx)
        super().generic_visit(stmt)

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        self.ctx.mesh_stack.append(stmt.mesh)
        try:
            # Fire any custom verify handler for MeshScope (none by default),
            # then recurse into body with the scope active.
            fn = self.registry.lookup(MeshScope)
            if fn is not None:
                fn(stmt, self.ctx)
            for child in stmt.body:
                self.visit(child)
        finally:
            self.ctx.mesh_stack.pop()


class CodegenVisitor:
    """Dual-path dispatch: Op (via Call) → str fragment; Stmt → emit into ctx.

    Not a subclass of StmtVisitor/ExprVisitor — codegen's two paths return
    different types (str for Op, None for Stmt) and need different entries.
    Uses `visit_<ClassName>` lookup style for API consistency with the rest
    of the visitor family.
    """

    def __init__(
        self,
        ctx,  # CodegenContext; concrete per-target type lives with the target
        target: str,
    ) -> None:
        self.ctx = ctx
        self.target = target
        self.registry = _codegen_registry_for(target)

    def emit_stmt(self, stmt: Stmt) -> None:
        fn = self.registry.lookup(type(stmt))
        if fn is None:
            raise RuntimeError(
                f"no @register_codegen_{self.target} for Stmt {type(stmt).__name__}"
            )
        fn(stmt, self.ctx)

    def emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, Call):
            fn = self.registry.lookup(type(expr.target))
            if fn is None:
                raise RuntimeError(
                    f"no @register_codegen_{self.target} for Op "
                    f"{type(expr.target).__name__}"
                )
            return fn(expr, self.ctx)
        # Leaf Expr nodes (Var / Constant) are emitted by the target's
        # CodegenContext helpers (e.g. ctx.name_for / ctx.literal). Callers
        # that want a generic fallback should override emit_expr.
        raise RuntimeError(
            f"CodegenVisitor.emit_expr: leaf Expr {type(expr).__name__} "
            "has no default emission; handle via target ctx helpers."
        )


class CostEvaluator(ExprVisitor[Cost]):
    """Dispatch the registered recursive-local Cost Evaluator per Op class.

    A missing evaluator fails closed — it is a construction error, not a
    zero-Cost default.
    """

    def __init__(
        self,
        ctx: CostContext,
        registry: AnalysisRegistry = cost_evaluator_registry,
    ) -> None:
        self.ctx = ctx
        self.registry = registry

    def visit_Call(self, call: Call) -> Cost:
        fn = self.registry.lookup(type(call.target))
        if fn is None:
            self.ctx.error(
                call, f"no cost evaluator registered for {type(call.target).__name__}"
            )
        return fn(call, self.ctx)


def _codegen_registry_for(target: str) -> AnalysisRegistry:
    if target == "cuda":
        return codegen_cuda_registry
    if target == "cpu":
        return codegen_cpu_registry
    raise ValueError(f"unknown codegen target: {target!r}")


__all__ = [
    "TypeInferVisitor",
    "VerifyVisitor",
    "CodegenVisitor",
    "CostEvaluator",
]
