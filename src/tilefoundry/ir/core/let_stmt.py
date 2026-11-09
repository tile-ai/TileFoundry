"""The let statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class LetStmt(Stmt):
    """Bind var to value for the following statements of the enclosing Sequential."""

    var: expr.Var
    value: expr.Expr

__all__ = ['LetStmt']
