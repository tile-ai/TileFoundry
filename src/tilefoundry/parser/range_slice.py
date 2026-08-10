"""Represent chunked tile iteration only while parsing.

Two-argument ``tile(extent, step)`` binds a range whose indexed use lowers to
``[iv * step, iv * step + step)`` while the IR retains its scalar induction
variable. Single-argument tile loops bind that scalar directly and never create
a range slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilefoundry.ir.core.expr import Constant, Expr, Var
from tilefoundry.ir.types.dim import DimAdd, DimMul, simplify_dim
from tilefoundry.ir.types.shape_helpers import i64_const


def _i64(value: int) -> Constant:
    return i64_const(value)


def _to_i64_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return _i64(value)
    raise TypeError(f"RangeSlice bound must be int or Expr, got {type(value).__name__}")


@dataclass(frozen=True)
class RangeSlice:
    """Bind a chunked tile loop's induction variable, extent, and step.

    ``induction_var`` is shared with the resulting grid region. ``extent`` is
    the complete range and ``step`` is the per-iteration chunk size.
    """

    induction_var: Var
    extent: Any
    step: Any

    @property
    def start(self) -> Expr:
        """Lower bound of the current iteration: ``iv * step``."""
        step_e = _to_i64_expr(self.step)
        return simplify_dim(DimMul, (self.induction_var, step_e))

    @property
    def stop(self) -> Expr:
        """Upper bound of the current iteration: ``iv * step + step``."""
        step_e = _to_i64_expr(self.step)
        return simplify_dim(DimAdd, (self.start, step_e))


__all__ = ["RangeSlice"]
