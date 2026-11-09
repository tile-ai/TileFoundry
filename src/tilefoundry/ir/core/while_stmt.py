"""The while loop statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class While(Stmt):
    cond: expr.Expr
    body: Sequential

__all__ = ['While']
