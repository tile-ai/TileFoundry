"""Traverse and identity-preservingly rewrite expression and statement IR.

Generic expression traversal excludes binding-site Vars. Statement traversal
keeps embedded expression rewriting explicit; function helpers enter through
the body while ``PrimFunction`` itself remains a Stmt.

See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable

from tilefoundry.ir import expr

from .evaluate import Evaluate
from .for_stmt import For
from .function import Function
from .if_stmt import If
from .let_stmt import LetStmt
from .loop_region import LoopRegion
from .mesh_region import MeshRegion
from .mesh_scope import MeshScope
from .module import ModuleList
from .prim_function import PrimFunction
from .return_stmt import Return
from .sequential import Sequential
from .shape_of import ShapeOf
from .stmt import Stmt
from .symbol_ref import SymbolRef
from .while_stmt import While

__all__ = [
    "TypeFunctor",
    "ExprVisitor",
    "ExprWalker",
    "ExprCollector",
    "collect_exprs",
    "ExprCloner",
    "BindingSubstitutionCloner",
    "expr_children",
    "function_values",
    "StmtVisitor",
    "StmtMutator",
    "StmtExprMutator",
    "walk_prim_function",
    "rewrite_prim_function",
]


def expr_children(node: expr.Expr) -> tuple[expr.Expr, ...]:
    """Enumerate direct child Expr nodes of `expr`.

    Binding-site Var fields (e.g. `LoopRegion.induction_var` /
    `LoopRegion.params`) are intentionally excluded — rewriting
    them with a generic ExprCloner could produce type-illegal nodes (a
    non-Var in a `tuple[Var, ...]` slot). A mutator that wants to rename
    or substitute bindings must override `visit_LoopRegion` and rebuild
    explicitly.
    """
    match node:
        case expr.Var() | expr.Constant() | SymbolRef() | ShapeOf() | ModuleList():
            return ()
        case expr.Call(target=target, args=args):
            return (target, *args) if isinstance(target, expr.Call) else args
        case MeshRegion(args=args, body=body):
            return (*args, body)
        case LoopRegion(args=args, body=body, yield_values=yield_values):
            return (*args, body, *yield_values)
        case Function(body=body):
            return (body,)
        case expr.Tuple(elements=elements):
            return elements
        case _:
            raise AssertionError(f"expr_children: unknown Expr subclass {type(node).__name__}")


def rebuild_expr(node: expr.Expr, new_children: tuple[expr.Expr, ...]) -> expr.Expr:
    """Rebuild `expr` with replaced children (same order as expr_children).

    Binding-site fields are carried over untouched.
    """
    match node:
        case expr.Var() | expr.Constant() | SymbolRef() | ModuleList():
            return node
        case expr.Call(target=expr.Call()):
            return replace(node, target=new_children[0], args=new_children[1:])
        case expr.Call():
            return replace(node, args=new_children)
        case MeshRegion(args=args):
            n_args = len(args)
            return replace(node, args=new_children[:n_args], body=new_children[n_args])
        case LoopRegion(args=args):
            n_init = len(args)
            init = new_children[:n_init]
            body = new_children[n_init]
            yields = new_children[n_init + 1 :]
            return replace(node, args=init, body=body, yield_values=yields)
        case Function():
            (body,) = new_children
            return replace(node, body=body)
        case expr.Tuple():
            return replace(node, elements=new_children)
        case _:
            raise AssertionError(f"rebuild_expr: unknown Expr subclass {type(node).__name__}")


def stmt_children(stmt: Stmt) -> tuple[Stmt, ...]:
    """Direct child Stmt nodes.

    Direct child Stmt nodes. (Not Expr fields — StmtVisitor does not descend
    into embedded Expr by default; see StmtExprMutator for combined traversal.)

    P2: ``body`` fields are ``Sequential`` (a Stmt), so control-flow /
    scope Stmts report their body as a single child Sequential.
    ``Sequential`` itself reports its packed ``body`` tuple as children.

    ``PrimFunction`` is itself a Stmt whose child is the body Sequential.

    See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
    """
    match stmt:
        case Sequential(body=body):
            return body
        case (
            PrimFunction(body=body)
            | For(body=body)
            | While(body=body)
            | MeshScope(body=body)
        ):
            return (body,)
        case If(then_body=then_body, else_body=else_body):
            return (then_body, else_body)
        case LetStmt() | Return() | Evaluate():
            return ()
        case _:
            raise AssertionError(f"stmt_children: unknown Stmt subclass {type(stmt).__name__}")


def rebuild_stmt_children(stmt: Stmt, new_children: tuple[Stmt, ...]) -> Stmt:
    """Replace the child Stmts of `stmt` (same order as stmt_children)."""
    match stmt:
        case Sequential():
            return replace(stmt, body=new_children)
        case PrimFunction() | For() | While() | MeshScope():
            (body,) = new_children
            assert isinstance(body, Sequential)
            return replace(stmt, body=body)
        case If():
            then_body, else_body = new_children
            assert isinstance(then_body, Sequential)
            assert isinstance(else_body, Sequential)
            return replace(stmt, then_body=then_body, else_body=else_body)
        case LetStmt() | Return() | Evaluate():
            return stmt
        case _:
            raise AssertionError(
                f"rebuild_stmt_children: unknown Stmt subclass {type(stmt).__name__}"
            )


def stmt_expr_fields(stmt: Stmt) -> tuple[str, ...]:
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
        case Evaluate(callable=SymbolRef() | expr.Call() | ModuleList()):
            return ("callable", "args")
        case Evaluate():
            return ("args",)

        case _:
            return ()


class TypeFunctor[T]:
    """Dispatch read-only operations over IR type and layout values.

    Types are immutable value objects rather than expressions, so they need a
    separate dispatch surface.  Keeping this visitor independent from the
    expression visitor prevents printers and analysis consumers from growing
    parallel ``isinstance`` ladders for the same type family.
    """

    def visit(self, value: Any, ctx: Any = None) -> T:
        return self.dispatch_visit(value, ctx)

    def dispatch_visit(self, value: Any, ctx: Any) -> T:
        method = getattr(self, f"visit_{type(value).__name__}", None)
        if method is not None:
            return method(value, ctx)
        return self.default_visit(value, ctx)

    def default_visit(self, value: Any, ctx: Any) -> T:
        raise NotImplementedError(f"no type visit routine for {type(value).__name__}")


class ExprVisitor[T](expr.Visitor[T]):
    """Read-only Expr traversal with identity-based DAG memoization over every node shape."""

    def __init__(
        self,
        *,
        memo: dict[int, tuple[expr.Expr, T]] | None = None,
        visit_other_functions: bool = False,
        root_function: expr.Expr | None = None,
    ) -> None:
        """Create a visitor with an identity memo and optional Function root.

        The Expr in each memo value pins the object that owns the integer key.
        Without that strong reference, Python may reuse an id after collection
        and return a result belonging to a different expression.
        """
        super().__init__(memo=memo)
        self.root_expr = root_function
        self.visit_other_functions = visit_other_functions

    def visit_operands(self, node: expr.Expr, ctx: Any) -> tuple[T, ...]:
        """Visit value children from the fixed `expr_children` table."""
        return tuple(self.visit(child, ctx) for child in expr_children(node))

    def can_visit_function_body(self, fn: expr.Expr) -> bool:
        """Whether the body of *fn* is entered: the root, or any function when asked to."""
        return self.visit_other_functions or fn is self.root_expr

    def visit_function_body(self, fn: expr.Expr, ctx: Any = None) -> T:
        """Enter `fn.body` while retaining `fn` as the explicit root."""
        if self.root_expr is None:
            self.root_expr = fn
        if not self.can_visit_function_body(fn):
            return None
        body = getattr(fn, "body", None)
        if body is None:
            raise ValueError(f"cannot visit body of prototype {fn!r}")
        return self.visit(body, ctx)


class ExprWalker[T](ExprVisitor[T]):
    """Memoized side-effect traversal for the known value Expr shapes."""

    @typing.override
    def default_visit_leaf(self, node: expr.Expr, operands: tuple[T, ...], ctx: Any) -> T:
        return None


class ExprCollector(ExprWalker[None]):
    """Append every visited Expr to the list passed as traversal context."""

    @typing.override
    def default_visit_leaf(
        self, node: expr.Expr, operands: tuple[None, ...], ctx: list[expr.Expr]
    ) -> None:
        ctx.append(node)


def collect_exprs(root: expr.Expr | None) -> tuple[expr.Expr, ...]:
    """Every value reachable from *root*, operands before their consumer.

    The body is an SSA DAG rather than a tree, so a shared value is visited
    once. Function reachability relies on this operand-before-consumer order:
    changing it changes the order of callees reached within one caller.
    """
    if root is None:
        return ()
    found: list[expr.Expr] = []
    ExprCollector().visit(root, found)
    return tuple(found)


def generic_expr_rewrite(node: expr.Expr, visit_fn: Callable[[expr.Expr], expr.Expr]) -> expr.Expr:
    """Rebuild `expr` from `visit_fn`-rewritten children, preserving identity when no child changed.

    Rebuild `expr` from `visit_fn`-rewritten children, preserving
    identity when no child changed. Shared by `ExprCloner.default_visit`
    and `StmtExprMutator.expr_generic_visit`.
    """
    children = expr_children(node)
    new_children = tuple(visit_fn(c) for c in children)
    if all(nc is oc for nc, oc in zip(new_children, children)):
        return node
    return rebuild_expr(node, new_children)


class ExprCloner(ExprVisitor[expr.Expr]):
    """Expr → Expr rewrite with identity preservation.

    Invariant: if every child visit returns an `is`-identical object, the
    original Expr is returned unchanged. This enables structure sharing and
    lets callers detect "did this pass change anything" via `new is old`.
    """

    def default_visit(self, node: expr.Expr, ctx: Any) -> expr.Expr:
        return generic_expr_rewrite(node, lambda child: self.visit(child, ctx))


def function_values(function: Function) -> tuple[expr.Expr, ...]:
    """The Function, its parameters, and every value reachable from its body."""
    return (function, *function.params, *collect_exprs(function.body))


class BindingSubstitutionCloner(ExprCloner):
    """Clone an Expr DAG while replacing Vars from an identity-keyed environment.

    This is an ``ExprCloner``, so one instance memoizes by source-expression
    identity: shared inputs remain shared in its output. Callers that need
    independent copies, such as two inline call sites, use one cloner instance
    per copy. Binding sites in a LoopRegion create an extended environment for
    its body and yields without rewriting a Var slot to a non-Var value.
    """

    def __init__(self, metadata: Callable[[expr.Expr], tuple] | None = None) -> None:
        super().__init__()
        self.cloned_metadata = metadata or (lambda node: node.metadata)

    def visit_Var(self, node: expr.Var, ctx: Mapping[int, expr.Expr]) -> expr.Expr:
        bound = ctx.get(id(node))
        if bound is not None:
            return bound
        return replace(node, metadata=self.cloned_metadata(node))

    def visit_Constant(self, node: expr.Constant, ctx: Mapping[int, expr.Expr]) -> expr.Expr:
        return replace(node, metadata=self.cloned_metadata(node))

    def visit_Tuple(self, node: expr.Tuple, ctx: Mapping[int, expr.Expr]) -> expr.Expr:
        return replace(
            node,
            elements=tuple(self.visit(item, ctx) for item in node.elements),
            metadata=self.cloned_metadata(node),
        )

    def visit_MeshRegion(self, node: MeshRegion, ctx: Mapping[int, expr.Expr]) -> expr.Expr:
        params = tuple(replace(param, metadata=self.cloned_metadata(param)) for param in node.params)
        inner = dict(ctx)
        inner.update((id(old), new) for old, new in zip(node.params, params, strict=True))
        return replace(
            node,
            params=params,
            args=tuple(self.visit(arg, ctx) for arg in node.args),
            body=self.visit(node.body, inner),
            metadata=self.cloned_metadata(node),
        )

    def visit_LoopRegion(self, region: LoopRegion, ctx: Mapping[int, expr.Expr]) -> LoopRegion:
        args = tuple(self.visit(item, ctx) for item in region.args)
        induction_var = replace(
            region.induction_var,
            metadata=self.cloned_metadata(region.induction_var),
        )
        params = tuple(
            replace(item, metadata=self.cloned_metadata(item)) for item in region.params
        )
        inner = dict(ctx)
        inner[id(region.induction_var)] = induction_var
        inner.update((id(old), new) for old, new in zip(region.params, params))
        return replace(
            region,
            start=(
                self.visit(region.start, ctx)
                if isinstance(region.start, expr.Expr)
                else region.start
            ),
            extent=(
                self.visit(region.extent, ctx)
                if isinstance(region.extent, expr.Expr)
                else region.extent
            ),
            step=(
                self.visit(region.step, ctx)
                if isinstance(region.step, expr.Expr)
                else region.step
            ),
            induction_var=induction_var,
            params=params,
            args=args,
            body=self.visit(region.body, inner),
            yield_values=tuple(self.visit(item, inner) for item in region.yield_values),
            metadata=self.cloned_metadata(region),
        )

    def visit_Call(self, node: expr.Call, ctx: Mapping[int, expr.Expr]) -> expr.Expr:
        return replace(
            node,
            target=self.visit(node.target, ctx) if isinstance(node.target, expr.Call) else node.target,
            args=tuple(
                self.visit(arg, ctx) if isinstance(arg, expr.Expr) else arg for arg in node.args
            ),
            metadata=self.cloned_metadata(node),
        )


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
        for child in stmt_children(stmt):
            self.visit(child)
        return None


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
        children = stmt_children(stmt)
        new_children = tuple(self.visit(c) for c in children)
        if all(nc is oc for nc, oc in zip(new_children, children)):
            return stmt
        return rebuild_stmt_children(stmt, new_children)


class StmtExprMutator(StmtMutator):
    """Rewrite statement structure and its embedded expression subtrees.

    Statements enter through ``visit`` and expressions through ``visit_expr``.
    Both use ``visit_<ClassName>`` overrides without mixing node categories.
    """

    def visit_stmt(self, stmt: Stmt) -> Stmt:
        return self.visit(stmt)

    def visit_expr(self, node: expr.Expr, ctx: Any = None) -> expr.Expr:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is not None:
            return method(node, ctx)
        return self.expr_generic_visit(node, ctx)

    def expr_generic_visit(self, node: expr.Expr, ctx: Any = None) -> expr.Expr:
        return generic_expr_rewrite(node, lambda child: self.visit_expr(child, ctx))

    def generic_visit(self, stmt: Stmt) -> Stmt:

        stmt_after_kids = StmtMutator.generic_visit(self, stmt)

        return rewrite_stmt_exprs(stmt_after_kids, lambda node: self.visit_expr(node, None))


def rewrite_stmt_exprs(stmt: Stmt, fn) -> Stmt:
    """Rewrite stmt exprs.

    Walk the Expr fields of `stmt`, rewrite each via `fn`, return a new
    Stmt if any changed, else the original (identity preservation).
    """
    field_names = stmt_expr_fields(stmt)
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
