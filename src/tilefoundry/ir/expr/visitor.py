"""Dispatch over expression nodes that know only ``Var``, ``Constant``, ``Call`` and ``Tuple``.

The structure-aware walkers of :mod:`tilefoundry.ir.core` extend these with the
nodes of functions, regions and statements.
"""

from __future__ import annotations

from typing import Any

from .expr import Call, Constant, Expr, Tuple, Var

__all__ = ["Functor", "Visitor"]


def children(expr: Expr) -> tuple[Expr, ...]:
    """The direct child nodes of a ``Var``, ``Constant``, ``Call`` or ``Tuple``."""
    match expr:
        case Var() | Constant():
            return ()
        case Call(target=target, args=args):
            return (target, *args) if isinstance(target, Call) else args
        case Tuple(elements=elements):
            return elements
        case _:
            raise AssertionError(f"children: unknown Expr subclass {type(expr).__name__}")


class Functor[T]:
    """Expr dispatch without memoization.

    Read-only visitors and identity-preserving mutators share this dispatch
    layer. The first expression visited is retained as a diagnostic root.
    """

    def __init__(self) -> None:
        """Start without a root."""
        self.root_expr: Expr | None = None

    def visit(self, expr: Expr, ctx: Any = None) -> T:
        """Dispatch on the node's class, retaining the first node as the root."""
        if self.root_expr is None:
            self.root_expr = expr
        return self.dispatch_visit(expr, ctx)

    def dispatch_visit(self, expr: Expr, ctx: Any) -> T:
        """Call ``visit_<ClassName>`` when the visitor defines it."""
        method = getattr(self, f"visit_{type(expr).__name__}", None)
        if method is not None:
            return method(expr, ctx)
        return self.default_visit(expr, ctx)

    def default_visit(self, expr: Expr, ctx: Any) -> T:
        """Refuse a node no ``visit_<ClassName>`` handles."""
        raise NotImplementedError(f"no visit routine for {type(expr).__name__}")

    def clear(self) -> None:
        """Forget the root."""
        self.root_expr = None


class Visitor[T](Functor[T]):
    """Read-only Expr traversal with identity-based DAG memoization."""

    def __init__(self, *, memo: dict[int, tuple[Expr, T]] | None = None) -> None:
        """Create a visitor with an identity memo.

        The Expr in each memo value pins the object that owns the integer key.
        Without that strong reference, Python may reuse an id after collection
        and return a result belonging to a different expression.
        """
        super().__init__()
        self.memo: dict[int, tuple[Expr, T]] = dict(memo) if memo else {}

    def dispatch_visit(self, expr: Expr, ctx: Any) -> T:
        """Visit *expr* once; a repeat visit returns the first result."""
        hit = self.memo.get(id(expr))
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
            elif type(self).default_visit_leaf is Visitor.default_visit_leaf:
                result = self.default_visit(expr, ctx)
            else:
                operands = self.visit_operands(expr, ctx)
                result = self.default_visit_leaf(expr, operands, ctx)
        self.memo[id(expr)] = (expr, result)
        return result

    def visit_operands(self, expr: Expr, ctx: Any) -> tuple[T, ...]:
        """Visit the value children of *expr*."""
        return tuple(self.visit(child, ctx) for child in children(expr))

    def default_visit_leaf(self, expr: Expr, operands: tuple[T, ...], ctx: Any) -> T:
        """The result for a node with no ``visit_leaf_<ClassName>``."""
        return self.default_visit(expr, ctx)

    def clear(self) -> None:
        """Forget the memo and the root."""
        self.memo.clear()
        super().clear()
