"""The expression-evaluation statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class Evaluate(Stmt):
    """Place a value-less effect Op or retained function invocation in statement position."""

    callable: "Op | FunctionRef | PrimFunction"
    args: tuple[expr.Expr, ...]

__all__ = ['Evaluate']
