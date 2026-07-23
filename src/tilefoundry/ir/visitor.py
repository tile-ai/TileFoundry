"""IR traversal / rewrite base classes (visitor + identity-preserving mutator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import (
    Abort,
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef

__all__ = [
    "ExprVisitor",
    "ExprMutator",
    "StmtVisitor",
    "StmtMutator",
    "StmtExprMutator",
    "walk_prim_function",
    "rewrite_prim_function",
]


# ---------------------------------------------------------------------------
# Expr children / rebuild tables
# ---------------------------------------------------------------------------


def _expr_children(expr: Expr) -> tuple[Expr, ...]:
    """Enumerate direct child Expr nodes of `expr`.

    Binding-site Var fields (e.g. `GridRegionExpr.induction_var` /
    `GridRegionExpr.carried_args`) are intentionally excluded — rewriting
    them with a generic ExprMutator could produce type-illegal nodes (a
    non-Var in a `tuple[Var, ...]` slot). A mutator that wants to rename
    or substitute bindings must override `visit_GridRegionExpr` and rebuild
    explicitly.
    """
    match expr:
        case Var() | Constant() | SymbolRef():
            return ()
        case Call(args=args):
            return args
        case GridRegionExpr(init_args=init_args, body=body, yield_values=yield_values):
            # ``init_args`` are value Exprs (the loop's initial carried values) and
            # are traversed / rewritten; ``induction_var`` / ``carried_args`` are
            # binding-site Vars and stay excluded.
            return (*init_args, body, *yield_values)
        case HirFunction(body=body):
            # ``Function.params`` are binding-site Vars (excluded from rewrite,
            # same rationale as GridRegionExpr.induction_var); ``return_type`` /
            # ``topologies`` are metadata, not Exprs. Only ``body`` is a child Expr.
            return (body,)
        case Tuple(elements=elements):
            return elements
        case _:
            raise AssertionError(f"_expr_children: unknown Expr subclass {type(expr).__name__}")


def _rebuild_expr(expr: Expr, new_children: tuple[Expr, ...]) -> Expr:
    """Rebuild `expr` with replaced children (same order as _expr_children).
    Binding-site fields are carried over untouched."""
    match expr:
        case Var() | Constant() | SymbolRef():
            return expr
        case Call():
            return replace(expr, args=new_children)
        case GridRegionExpr(init_args=init_args):
            n_init = len(init_args)
            init = new_children[:n_init]
            body = new_children[n_init]
            yields = new_children[n_init + 1 :]
            return replace(expr, init_args=init, body=body, yield_values=yields)
        case HirFunction():
            (body,) = new_children
            return replace(expr, body=body)
        case Tuple():
            return replace(expr, elements=new_children)
        case _:
            raise AssertionError(f"_rebuild_expr: unknown Expr subclass {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Stmt children / Expr-field enumeration tables
# ---------------------------------------------------------------------------


def _stmt_children(stmt: Stmt) -> tuple[Stmt, ...]:
    """Direct child Stmt nodes. (Not Expr fields — StmtVisitor does not descend
    into embedded Expr by default; see StmtExprMutator for combined traversal.)

    P2: ``body`` fields are ``Sequential`` (a Stmt), so control-flow /
    scope / binding Stmts report their body as a single child Sequential.
    ``Sequential`` itself reports its packed ``body`` tuple as children.

    Per tir.md §2 ``PrimFunction`` is itself a Stmt — its single child is
    the body Sequential.
    """
    match stmt:
        case Sequential(body=body):
            return body
        case (
            PrimFunction(body=body)
            | LetStmt(body=body)
            | For(body=body)
            | While(body=body)
            | MeshScope(body=body)
        ):
            return (body,)
        case If(then_body=then_body, else_body=else_body):
            return (then_body, else_body)
        case DispatchCall(case_calls=case_calls, fallback=fallback):
            # case_calls are Evaluate(SymbolRef) (leaf Stmts); fallback is a Sequential.
            return (*case_calls, fallback)
        # Leaf-in-stmt-tree: no nested Stmt.
        case Return() | Evaluate() | Abort():
            return ()
        case _:
            raise AssertionError(f"_stmt_children: unknown Stmt subclass {type(stmt).__name__}")


def _rebuild_stmt_children(stmt: Stmt, new_children: tuple[Stmt, ...]) -> Stmt:
    """Replace the child Stmts of `stmt` (same order as _stmt_children)."""
    match stmt:
        case Sequential():
            return replace(stmt, body=new_children)
        case PrimFunction() | LetStmt() | For() | While() | MeshScope():
            (body,) = new_children
            assert isinstance(body, Sequential)
            return replace(stmt, body=body)
        case If():
            then_body, else_body = new_children
            assert isinstance(then_body, Sequential)
            assert isinstance(else_body, Sequential)
            return replace(stmt, then_body=then_body, else_body=else_body)
        case DispatchCall():
            *new_case_calls, new_fallback = new_children
            for nc in new_case_calls:
                assert isinstance(nc, Evaluate)
            assert isinstance(new_fallback, Sequential)
            return replace(
                stmt,
                case_calls=tuple(new_case_calls),
                fallback=new_fallback,
            )
        case Return() | Evaluate() | Abort():
            return stmt
        case _:
            raise AssertionError(
                f"_rebuild_stmt_children: unknown Stmt subclass {type(stmt).__name__}"
            )


def _stmt_expr_fields(stmt: Stmt) -> tuple[str, ...]:
    """Names of Expr-typed fields on `stmt`. StmtExprMutator uses this to
    rewrite the Expr subtrees embedded inside a Stmt. Var-binding fields
    (For.induction_var, LetStmt.var, MeshScope.binding) are intentionally
    excluded — a rewrite must not turn a binding site into a non-Var."""
    match stmt:
        case LetStmt():
            return ("value",)
        case For():
            return ("start", "stop", "step")
        case While() | If():
            return ("cond",)
        case Evaluate(callable=SymbolRef()):
            # Evaluate's embedded Exprs are its args; the callable is an Op (not
            # an Expr) unless it is a SymbolRef, which is then exposed too.
            return ("callable", "args")
        case Evaluate():
            return ("args",)
        # Sequential / Return / MeshScope: no embedded Expr to rewrite.
        case _:
            return ()


# ---------------------------------------------------------------------------
# Expr visitor / mutator
# ---------------------------------------------------------------------------


class ExprVisitor[T]:
    """Read-only Expr traversal. Override visit_<ClassName> to inject logic."""

    def visit(self, expr: Expr) -> T:
        method = getattr(self, f"visit_{type(expr).__name__}", None)
        if method is not None:
            return method(expr)
        return self.generic_visit(expr)

    def generic_visit(self, expr: Expr) -> T:
        """Default: recurse into all children; return None. Subclasses may
        override to aggregate."""
        for child in _expr_children(expr):
            self.visit(child)
        return None  # type: ignore[return-value]


def _dispatch(obj: Any, node: Any, generic: Callable[[Any], Any]) -> Any:
    """`visit_<type(node).__name__>` dispatch, falling back to `generic`.

    Shared by every `visit` / `visit_expr` entry point in this module —
    only the fallback (a `generic_visit`-shaped bound method) differs
    per caller.
    """
    method = getattr(obj, f"visit_{type(node).__name__}", None)
    if method is not None:
        return method(node)
    return generic(node)


def _generic_expr_rewrite(expr: Expr, visit_fn: Callable[[Expr], Expr]) -> Expr:
    """Rebuild `expr` from `visit_fn`-rewritten children, preserving
    identity when no child changed. Shared by `ExprMutator.generic_visit`
    and `StmtExprMutator._expr_generic_visit`."""
    children = _expr_children(expr)
    new_children = tuple(visit_fn(c) for c in children)
    if all(nc is oc for nc, oc in zip(new_children, children)):
        return expr
    return _rebuild_expr(expr, new_children)


class ExprMutator:
    """Expr → Expr rewrite with identity preservation.

    Invariant: if every child visit returns an `is`-identical object, the
    original Expr is returned unchanged. This enables structure sharing and
    lets callers detect "did this pass change anything" via `new is old`.
    """

    def visit(self, expr: Expr) -> Expr:
        return _dispatch(self, expr, self.generic_visit)

    def generic_visit(self, expr: Expr) -> Expr:
        return _generic_expr_rewrite(expr, self.visit)


# ir.hir.function imports ExprMutator (for its elaboration mutator) at
# module level, so this module-level import is positioned after
# ExprMutator is defined: whichever of the two modules loads first, the
# other's back-reference finds an already-bound name instead of hitting a
# partially-initialized module.
from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: E402

# ---------------------------------------------------------------------------
# Stmt visitor / mutator
# ---------------------------------------------------------------------------


class StmtVisitor[T]:
    """Read-only Stmt traversal. Does NOT descend into embedded Expr fields
    (use StmtExprMutator if you need Expr-level rewriting too)."""

    def visit(self, stmt: Stmt) -> T:
        method = getattr(self, f"visit_{type(stmt).__name__}", None)
        if method is not None:
            return method(stmt)
        return self.generic_visit(stmt)

    def generic_visit(self, stmt: Stmt) -> T:
        for child in _stmt_children(stmt):
            self.visit(child)
        return None  # type: ignore[return-value]


class StmtMutator:
    """Stmt → Stmt rewrite with identity preservation. Does not rewrite
    embedded Expr fields by default."""

    def visit(self, stmt: Stmt) -> Stmt:
        method = getattr(self, f"visit_{type(stmt).__name__}", None)
        if method is not None:
            return method(stmt)
        return self.generic_visit(stmt)

    def generic_visit(self, stmt: Stmt) -> Stmt:
        children = _stmt_children(stmt)
        new_children = tuple(self.visit(c) for c in children)
        if all(nc is oc for nc, oc in zip(new_children, children)):
            return stmt
        return _rebuild_stmt_children(stmt, new_children)


class StmtExprMutator(StmtMutator):
    """Rewrites both the Stmt tree structure AND the Expr subtrees embedded
    inside Stmts.

    Stmt path: `visit(stmt)` → `visit_<StmtClass>` override or `generic_visit`.
    Expr path: `visit_expr(expr)` → `visit_<ExprClass>` override or internal
    Expr generic visit.

    Both paths share the `visit_<ClassName>` convention on `self`, but are
    routed through different entry methods so Stmt-shaped and Expr-shaped
    nodes don't collide through shared `visit` dispatch.
    """

    def visit_stmt(self, stmt: Stmt) -> Stmt:
        return self.visit(stmt)

    def visit_expr(self, expr: Expr) -> Expr:
        return _dispatch(self, expr, self._expr_generic_visit)

    def _expr_generic_visit(self, expr: Expr) -> Expr:
        return _generic_expr_rewrite(expr, self.visit_expr)

    def generic_visit(self, stmt: Stmt) -> Stmt:  # type: ignore[override]
        # First rewrite child Stmts (StmtMutator identity rule).
        stmt_after_kids = StmtMutator.generic_visit(self, stmt)
        # Then rewrite embedded Expr fields on the (possibly new) Stmt.
        return _rewrite_stmt_exprs(stmt_after_kids, self.visit_expr)


def _rewrite_stmt_exprs(stmt: Stmt, fn) -> Stmt:
    """Walk the Expr fields of `stmt`, rewrite each via `fn`, return a new
    Stmt if any changed, else the original (identity preservation)."""
    field_names = _stmt_expr_fields(stmt)
    if not field_names:
        return stmt
    updates: dict[str, object] = {}
    changed = False
    for name in field_names:
        old = getattr(stmt, name)
        if isinstance(old, tuple):
            new_tup = tuple(fn(e) for e in old)
            if any(ne is not oe for ne, oe in zip(new_tup, old)):
                updates[name] = new_tup
                changed = True
        else:
            new = fn(old)
            if new is not old:
                updates[name] = new
                changed = True
    if not changed:
        return stmt
    return replace(stmt, **updates)


# ---------------------------------------------------------------------------
# Function-level helpers
#
# PrimFunction is itself a Stmt (tir.md §2); these helpers remain as the
# canonical entry points for per-function traversal so pass code doesn't
# have to distinguish "call visit(pf) vs walk inside body".
# ---------------------------------------------------------------------------


def walk_prim_function(visitor: StmtVisitor, pf: PrimFunction) -> None:
    """Apply `visitor` to ``pf.body`` (a ``Sequential``). Read-only."""
    visitor.visit(pf.body)


def rewrite_prim_function(mutator: StmtMutator, pf: PrimFunction) -> PrimFunction:
    """Rewrite ``pf.body`` through ``mutator``. Returns the original ``pf``
    when the rewritten Sequential is identity-equal to the original."""
    new_body = mutator.visit(pf.body)
    if new_body is pf.body:
        return pf
    assert isinstance(new_body, Sequential)
    return replace(pf, body=new_body)
