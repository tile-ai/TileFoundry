"""Parser-side ``RangeSlice`` for chunked tile iteration.


A ``RangeSlice`` represents the per-iteration sub-range yielded by a
two-arg ``tile(extent, step)`` form. It exists only at parse time —
it does **not** appear in the IR. The IR-level induction variable
(``Var(i64)``) is still emitted as ``GridRegionExpr.induction_var``;
the RangeSlice merely lets ``x[:, ok]`` lift to a ``Slice`` Op call
whose bounds are ``[iv * step, iv * step + step)``.

Example::

    for ok in tile(2048, 512):
        x_smem = reshard(x[:, ok], ...)   # x[:, ok*512 : ok*512+512]

For single-arg ``tile(extent)`` the loop iv is bound directly to the
``Var(i64)`` (legacy scalar form); ``RangeSlice`` is not used.
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
    """Parser-side iter binding for ``for ok in tile(extent, step)``.

    Attributes
    ----------
    induction_var : Var
        The IR i64 scalar Var emitted as the GridRegionExpr induction
        variable. Shared with ``GridRegionExpr.induction_var``.
    extent : Expr | int
        Total range covered across all iterations.
    step : Expr | int
        Per-iteration chunk size.
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
