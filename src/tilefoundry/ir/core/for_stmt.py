"""The counted loop statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class For(Stmt):
    induction_var: expr.Var
    start: expr.Expr
    stop: expr.Expr
    step: expr.Expr
    body: Sequential

__all__ = ['For']
