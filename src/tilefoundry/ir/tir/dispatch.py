"""``tir.DispatchCall`` — pattern-based first-match dispatch op.

The i-th ``case_patterns`` matches against ``subjects`` (by position);
the first matching case executes ``case_calls[i]``. No match runs
``fallback``. Source order is part of the IR contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr
from tilefoundry.ir.core.pattern import Pattern
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Evaluate, Sequential


@dataclass(frozen=True)
class DispatchCall(Stmt):
    """Dispatch by first matching case in source order.

    ``case_calls`` contains one evaluated symbol call per pattern tuple.
    """

    callee_name: str
    subjects: tuple[Expr, ...]
    case_patterns: tuple[tuple[Pattern, ...], ...]
    case_calls: tuple[Evaluate, ...]
    fallback: Sequential


__all__ = ["DispatchCall"]
