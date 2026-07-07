from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.types.shape_dim import ShapeDim


@dataclass(frozen=True)
class GridRegionExpr(Expr):
    """Loop-phi-shaped structured SSA folding a tile-style loop into one Expr."""

    induction_var: Var
    carried_args: tuple[Var, ...]
    init_args: tuple[Expr, ...]
    body: Expr
    yield_values: tuple[Expr, ...]
    extent: ShapeDim
    step: ShapeDim
    start: ShapeDim = 0


__all__ = ["GridRegionExpr"]
