"""Traverse and identity-preservingly rewrite expression and statement IR.

Generic expression traversal excludes binding-site Vars. Statement traversal
keeps embedded expression rewriting explicit; function helpers enter through
the body while ``PrimFunction`` itself remains a Stmt.

See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
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
    "ExprFunctor",
    "ExprVisitor",
    "ExprWalker",
    "ExprCollector",
    "collect_exprs",
    "ExprCloner",
    "StmtVisitor",
    "StmtMutator",
    "StmtExprMutator",
    "walk_prim_function",
    "rewrite_prim_function",
]


def _expr_children(expr: Expr) -> tuple[Expr, ...]:
    """Enumerate direct child Expr nodes of `expr`.

    Binding-site Var fields (e.g. `GridRegionExpr.induction_var` /
    `GridRegionExpr.carried_args`) are intentionally excluded — rewriting
    them with a generic ExprCloner could produce type-illegal nodes (a
    non-Var in a `tuple[Var, ...]` slot). A mutator that wants to rename
    or substitute bindings must override `visit_GridRegionExpr` and rebuild
    explicitly.
    """
    match expr:
        case Var() | Constant() | SymbolRef() | ShapeOf():
            return ()
        case Call(args=args):
            return args
        case GridRegionExpr(init_args=init_args, body=body, yield_values=yield_values):
            return (*init_args, body, *yield_values)
        case HirFunction(body=body):
            return (body,)
        case Tuple(elements=elements):
            return elements
        case _:
            raise AssertionError(f"_expr_children: unknown Expr subclass {type(expr).__name__}")


def _rebuild_expr(expr: Expr, new_children: tuple[Expr, ...]) -> Expr:
    """Rebuild `expr` with replaced children (same order as _expr_children).

    Binding-site fields are carried over untouched.
    """
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


def _stmt_children(stmt: Stmt) -> tuple[Stmt, ...]:
    """Direct child Stmt nodes.

    Direct child Stmt nodes. (Not Expr fields — StmtVisitor does not descend
    into embedded Expr by default; see StmtExprMutator for combined traversal.)

    P2: ``body`` fields are ``Sequential`` (a Stmt), so control-flow /
    scope / binding Stmts report their body as a single child Sequential.
    ``Sequential`` itself reports its packed ``body`` tuple as children.

    ``PrimFunction`` is itself a Stmt whose child is the body Sequential.

    See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
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
            return (*case_calls, fallback)

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
    """Names of Expr-typed fields on `stmt`.

    Names of Expr-typed fields on `stmt`. StmtExprMutator uses this to
    rewrite the Expr subtrees embedded inside a Stmt. Var-binding fields
    (For.induction_var, LetStmt.var, MeshScope.binding) are intentionally
    excluded — a rewrite must not turn a binding site into a non-Var.
    """
    match stmt:
        case LetStmt():
            return ("value",)
        case For():
            return ("start", "stop", "step")
        case While() | If():
            return ("cond",)
        case Evaluate(callable=SymbolRef()):
            return ("callable", "args")
        case Evaluate():
            return ("args",)

        case _:
            return ()


class ExprFunctor[T]:
    """Expr dispatch without memoization.

    Read-only visitors and identity-preserving mutators share this dispatch
    layer. The first expression visited is retained as a diagnostic root; a
    caller that enters through a function body must provide the Function root
    explicitly through ``ExprVisitor(root_function=...)``.
    """

    def __init__(self) -> None:
        self._root: Expr | None = None

    def visit(self, expr: Expr, ctx: Any = None) -> T:
        if self._root is None:
            self._root = expr
        return self.dispatch_visit(expr, ctx)

    def dispatch_visit(self, expr: Expr, ctx: Any) -> T:
        method = getattr(self, f"visit_{type(expr).__name__}", None)
        if method is not None:
            return method(expr, ctx)
        return self.default_visit(expr, ctx)

    def default_visit(self, expr: Expr, ctx: Any) -> T:
        raise NotImplementedError(f"no visit routine for {type(expr).__name__}")

    def clear(self) -> None:
        self._root = None


class ExprVisitor[T](ExprFunctor[T]):
    """Read-only Expr traversal with identity-based DAG memoization."""

    def __init__(
        self,
        *,
        memo: dict[int, tuple[Expr, T]] | None = None,
        visit_other_functions: bool = False,
        root_function: Expr | None = None,
    ) -> None:
        """Create a visitor with an identity memo and optional Function root.

        The Expr in each memo value pins the object that owns the integer key.
        Without that strong reference, Python may reuse an id after collection
        and return a result belonging to a different expression.
        """
        super().__init__()
        self._root = root_function
        self._visit_other_functions = visit_other_functions
        self._memo: dict[int, tuple[Expr, T]] = dict(memo) if memo else {}

    def dispatch_visit(self, expr: Expr, ctx: Any) -> T:
        hit = self._memo.get(id(expr))
        if hit is not None:
            return hit[1]
        leaf = getattr(self, f"visit_leaf_{type(expr).__name__}", None)
        if leaf is not None:
            operands = self.visit_operands(expr, ctx)
            result = leaf(expr, operands, ctx)
        else:
            legacy = getattr(self, f"visit_{type(expr).__name__}", None)
            if legacy is not None:
                result = legacy(expr, ctx)
            elif type(self).default_visit_leaf is ExprVisitor.default_visit_leaf:
                result = self.default_visit(expr, ctx)
            else:
                operands = self.visit_operands(expr, ctx)
                result = self.default_visit_leaf(expr, operands, ctx)
        self._memo[id(expr)] = (expr, result)
        return result

    def visit_operands(self, expr: Expr, ctx: Any) -> tuple[T, ...]:
        """Visit value children from the fixed `_expr_children` table."""
        return tuple(self.visit(child, ctx) for child in _expr_children(expr))

    def default_visit_leaf(self, expr: Expr, operands: tuple[T, ...], ctx: Any) -> T:
        return self.default_visit(expr, ctx)

    def can_visit_function_body(self, fn: Expr) -> bool:
        return self._visit_other_functions or fn is self._root

    def visit_function_body(self, fn: Expr, ctx: Any = None) -> T:
        """Enter `fn.body` while retaining `fn` as the explicit root."""
        if self._root is None:
            self._root = fn
        if not self.can_visit_function_body(fn):
            return None  # type: ignore[return-value]
        body = getattr(fn, "body", None)
        if body is None:
            raise ValueError(f"cannot visit body of prototype {fn!r}")
        return self.visit(body, ctx)

    def clear(self) -> None:
        self._memo.clear()
        super().clear()


class ExprWalker[T](ExprVisitor[T]):
    """Memoized side-effect traversal for the known value Expr shapes."""

    def default_visit_leaf(self, expr: Expr, operands: tuple[T, ...], ctx: Any) -> T:
        return None  # type: ignore[return-value]


class ExprCollector(ExprWalker[None]):
    """Append every visited Expr to the list passed as traversal context."""

    def default_visit_leaf(
        self, expr: Expr, operands: tuple[None, ...], ctx: list[Expr]
    ) -> None:
        ctx.append(expr)


def collect_exprs(root: Expr | None) -> tuple[Expr, ...]:
    """Every value reachable from *root*, operands before their consumer.

    The body is an SSA DAG rather than a tree, so a shared value is visited
    once. Function reachability relies on this operand-before-consumer order:
    changing it changes the order of callees reached within one caller.
    """
    if root is None:
        return ()
    found: list[Expr] = []
    ExprCollector().visit(root, found)
    return tuple(found)


def _generic_expr_rewrite(expr: Expr, visit_fn: Callable[[Expr], Expr]) -> Expr:
    """Rebuild `expr` from `visit_fn`-rewritten children, preserving identity when no child changed.

    Rebuild `expr` from `visit_fn`-rewritten children, preserving
    identity when no child changed. Shared by `ExprCloner.default_visit`
    and `StmtExprMutator._expr_generic_visit`.
    """
    children = _expr_children(expr)
    new_children = tuple(visit_fn(c) for c in children)
    if all(nc is oc for nc, oc in zip(new_children, children)):
        return expr
    return _rebuild_expr(expr, new_children)


class ExprCloner(ExprVisitor[Expr]):
    """Expr → Expr rewrite with identity preservation.

    Invariant: if every child visit returns an `is`-identical object, the
    original Expr is returned unchanged. This enables structure sharing and
    lets callers detect "did this pass change anything" via `new is old`.
    """

    def default_visit(self, expr: Expr, ctx: Any) -> Expr:
        return _generic_expr_rewrite(expr, lambda child: self.visit(child, ctx))


from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: E402


class StmtVisitor[T]:
    """Read-only Stmt traversal.

    Read-only Stmt traversal. Does NOT descend into embedded Expr fields
    (use StmtExprMutator if you need Expr-level rewriting too).
    """

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
    """Stmt → Stmt rewrite with identity preservation.

    Stmt → Stmt rewrite with identity preservation. Does not rewrite
    embedded Expr fields by default.
    """

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
    """Rewrite statement structure and its embedded expression subtrees.

    Statements enter through ``visit`` and expressions through ``visit_expr``.
    Both use ``visit_<ClassName>`` overrides without mixing node categories.
    """

    def visit_stmt(self, stmt: Stmt) -> Stmt:
        return self.visit(stmt)

    def visit_expr(self, expr: Expr, ctx: Any = None) -> Expr:
        method = getattr(self, f"visit_{type(expr).__name__}", None)
        if method is not None:
            return method(expr, ctx)
        return self._expr_generic_visit(expr, ctx)

    def _expr_generic_visit(self, expr: Expr, ctx: Any = None) -> Expr:
        return _generic_expr_rewrite(expr, lambda child: self.visit_expr(child, ctx))

    def generic_visit(self, stmt: Stmt) -> Stmt:  # type: ignore[override]

        stmt_after_kids = StmtMutator.generic_visit(self, stmt)

        return _rewrite_stmt_exprs(stmt_after_kids, lambda expr: self.visit_expr(expr, None))


def _rewrite_stmt_exprs(stmt: Stmt, fn) -> Stmt:
    """Rewrite stmt exprs.

    Walk the Expr fields of `stmt`, rewrite each via `fn`, return a new
    Stmt if any changed, else the original (identity preservation).
    """
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


def walk_prim_function(visitor: StmtVisitor, pf: PrimFunction) -> None:
    """Apply *visitor* read-only to a ``PrimFunction`` body."""
    visitor.visit(pf.body)


def rewrite_prim_function(mutator: StmtMutator, pf: PrimFunction) -> PrimFunction:
    """Rewrite a ``PrimFunction`` body, preserving identity when unchanged."""
    new_body = mutator.visit(pf.body)
    if new_body is pf.body:
        return pf
    assert isinstance(new_body, Sequential)
    return replace(pf, body=new_body)
