"""The conditional statement."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class If(Stmt):
    cond: expr.Expr
    then_body: Sequential
    else_body: Sequential

__all__ = ['If']
