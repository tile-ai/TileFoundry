"""Derived Visitors — TypeInferVisitor / VerifyVisitor / CodegenVisitor / CostEvaluator.

`AnalysisRegistry` instance with a traversal skeleton from
tilefoundry.ir.visitor.

The `registry` is exposed as an advanced constructor param (default: the
canonical module-level registry for that analysis). Default path uses the
module-level registry directly; passing a custom one is an advanced
extension point for sandbox tests or grouped dispatch.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core.expr import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope
from tilefoundry.ir.types.substitute import canonicalize_dims
from tilefoundry.ir.types.tensor_type import TensorType, TupleType, Type, UnitType
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.ir.visitor import ExprVisitor, ExprWalker, StmtVisitor

from .contexts import Cost, CostContext, TypeInferContext, VerifyContext
from .registries import (
    AnalysisRegistry,
    cost_evaluator_registry,
    typeinfer_registry,
    verify_stmt_registry,
)


class TypeInferVisitor(ExprVisitor[Type]):
    """Derive one type for each ``Expr`` kind.

    See [hir §1.1](docs/spec/hir.md#11-function) and
    [visitor-registry §4](docs/spec/visitor-registry.md#4-instance-1--typeinfer).

    The context owns one identity memo for the current inference scope. A
    missing leaf raises through ``default_visit_leaf`` rather than trusting a
    stale ``expr.type``.
    """

    def __init__(self, *, memo=None, owns_body: bool = True) -> None:
        super().__init__(memo=memo)
        self._memo_supplied = memo is not None
        self._visit_depth = 0
        self._owns_body = owns_body

    def visit(self, expr: Expr, ctx: TypeInferContext) -> Type:
        outermost = self._visit_depth == 0
        if outermost:
            if self._memo_supplied:
                ctx = replace(ctx, memo=self._memo)
            else:
                self._memo = ctx.memo
        self._visit_depth += 1
        try:
            result = canonicalize_dims(super().visit(expr, ctx))
            if self._owns_body:
                expr.type = result
            return result
        finally:
            self._visit_depth -= 1

    def visit_leaf_Var(self, var: Var, _operands, ctx: TypeInferContext) -> Type:
        return var.annotation

    def visit_leaf_Constant(self, c: Constant, _operands, ctx: TypeInferContext) -> Type:
        return c.type

    def visit_leaf_Call(self, call: Call, arg_types, ctx: TypeInferContext) -> Type:
        target = call.target
        if isinstance(target, Function):
            return self._call_function(call, target, arg_types, ctx)
        op_cls = type(target)
        fn = typeinfer_registry.lookup(op_cls)
        if fn is None:
            ctx.error(call, f"no typeinfer registered for {op_cls.__name__}")
        return fn(call, ctx)

    def _call_function(
        self,
        call: Call,
        callee: Function,
        arg_types: tuple[Type, ...],
        ctx: TypeInferContext,
    ) -> Type:
        child = ctx.child_for(callee)
        supplied = tuple(
            param for param in callee.params if not (child is not None and param.is_const)
        )
        if len(arg_types) != len(supplied):
            kind = "activation(s)" if child is not None else "parameter(s)"
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arity mismatch — "
                f"callee declares {len(supplied)} {kind}, call passed {len(arg_types)}",
            )

        given = iter(enumerate(arg_types))
        memo = {}
        for param in callee.params:
            if child is not None and param.is_const:
                memo[id(param)] = (param, param.annotation)
                continue
            index, arg_type = next(given)
            declared = param.annotation
            wildcard = isinstance(declared, TensorType) and declared.layout is None
            if not types_compatible(declared, arg_type):
                if wildcard and isinstance(arg_type, TensorType):
                    message = (
                        f"hir Function call {callee.name!r}: arg {index} shape/dtype "
                        f"mismatch — callee param {param.name!r} expects logical "
                        f"{declared.shape} {declared.dtype.name}, got "
                        f"{arg_type.shape} {arg_type.dtype.name}"
                    )
                else:
                    message = (
                        f"hir Function call {callee.name!r}: arg {index} type mismatch — "
                        f"callee param {param.name!r} expects {declared!r}, got {arg_type!r}"
                    )
                ctx.error(call, message)
            bound = arg_type if wildcard else declared
            memo[id(param)] = (param, bound)

        if callee.body is None or callee.variants:
            return callee.return_type

        key = (id(callee), arg_types)
        cached = ctx.instantiated_memo.get(key)
        if cached is not None:
            return cached
        result = TypeInferVisitor(memo=memo, owns_body=False).visit(
            callee.body, ctx.for_callee(callee)
        )
        ctx.instantiated_memo[key] = result
        return result

    def visit_leaf_Tuple(self, tup: Tuple, operands, ctx: TypeInferContext) -> Type:
        """Visit Tuple.

        Structural: the field types of the (possibly just-elaborated)
        elements, never the node's own stamped ``.type`` ([hir §1.1](docs/spec/hir.md#11-function)).
        """
        return TupleType(fields=operands)

    def visit_GridRegionExpr(self, grid: GridRegionExpr, ctx: TypeInferContext) -> Type:
        """Infer a loop after binding its induction and carried variables."""
        inits = tuple(self.visit(arg, ctx) for arg in grid.init_args)
        memo = {
            **self._memo,
            id(grid.induction_var): (grid.induction_var, grid.induction_var.annotation),
            **{id(phi): (phi, type_) for phi, type_ in zip(grid.carried_args, inits)},
        }
        inner = TypeInferVisitor(memo=memo, owns_body=self._owns_body)
        body_type = inner.visit(grid.body, ctx)
        for y in grid.yield_values:
            inner.visit(y, ctx)
        if not grid.carried_args:
            return body_type
        if len(grid.carried_args) == 1:
            return inner.visit(grid.carried_args[0], ctx)
        return TupleType(fields=tuple(inner.visit(phi, ctx) for phi in grid.carried_args))

    def visit_leaf_ShapeOf(
        self, shape_of: ShapeOf, _operands, ctx: TypeInferContext
    ) -> Type:
        """A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32) type at construction.

        A ``tir.ShapeOf`` always carries its own concrete (rank-0 i32)
        type at construction; it has no children to derive from.
        """
        return shape_of.type

    def default_visit_leaf(self, expr: Expr, _operands, ctx: TypeInferContext) -> Type:
        ctx.error(expr, f"no typeinfer rule for Expr subclass {type(expr).__name__}")


def inference_type(expr: Expr, ctx: TypeInferContext | None = None) -> Type:
    """Infer and return *expr*'s type without writing it back to the IR."""
    return TypeInferVisitor(owns_body=False).visit(
        expr, ctx if ctx is not None else TypeInferContext()
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
        registry: AnalysisRegistry = verify_stmt_registry,
    ) -> None:




        self.ctx = ctx
        self.registry = registry

    def generic_visit(self, stmt: Stmt) -> None:
        if isinstance(stmt, Evaluate):



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
        ctx,
        registry: AnalysisRegistry,
        *,
        backend: str,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.backend = backend
        self.registry = registry

    def emit_stmt(self, stmt: Stmt) -> None:
        fn = self.registry.lookup(type(stmt))
        if fn is None:
            raise RuntimeError(
                f"no @register_codegen_{self.backend} for Stmt "
                f"{type(stmt).__name__}"
            )
        fn(stmt, self.ctx)

    def emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, Call):
            fn = self.registry.lookup(type(expr.target))
            if fn is None:
                raise RuntimeError(
                    f"no @register_codegen_{self.backend} for Op "
                    f"{type(expr.target).__name__}"
                )
            return fn(expr, self.ctx)



        raise RuntimeError(
            f"CodegenVisitor.emit_expr: leaf Expr {type(expr).__name__} "
            "has no default emission; handle via target ctx helpers."
        )


class CostEvaluator(ExprWalker[Cost]):
    """Dispatch the registered recursive-local Cost Evaluator per Op class.

    A missing evaluator fails closed — it is a construction error, not a
    zero-Cost default.
    """

    def __init__(
        self,
        registry: AnalysisRegistry = cost_evaluator_registry,
    ) -> None:
        super().__init__()
        self.registry = registry

    def visit_Call(self, call: Call, ctx: CostContext) -> Cost:
        fn = self.registry.lookup(type(call.target))
        if fn is None:
            ctx.error(
                call, f"no cost evaluator registered for {type(call.target).__name__}"
            )
        return fn(call, ctx)

__all__ = [
    "TypeInferVisitor",
    "inference_type",
    "VerifyVisitor",
    "CodegenVisitor",
    "CostEvaluator",
]
