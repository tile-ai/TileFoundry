"""The ``LoopRegion`` expression: a tile loop as one structured SSA region."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr, types


@dataclass(unsafe_hash=True)
class LoopRegion(expr.Expr):
    """Loop-phi-shaped structured SSA folding a tile-style loop into one region.

    A region rather than a scope: the values the loop carries are its own state,
    stated once here, not something every expression inside holds a copy of.

    The first len(yield_values) params carry state; remaining params capture
    invariant values. All matching args are evaluated outside the loop.
    """

    induction_var: expr.Var
    params: tuple[expr.Var, ...]
    args: tuple[expr.Expr, ...]
    body: expr.Expr
    yield_values: tuple[expr.Expr, ...]
    extent: types.ShapeDim
    step: types.ShapeDim
    start: types.ShapeDim = 0

    def carries(self):
        return tuple(zip(self.params, self.args, self.yield_values))

    def captures(self) -> tuple[tuple[expr.Var, expr.Expr], ...]:
        """Pair invariant parameters with their arguments, excluding carries."""
        carried = len(self.yield_values)
        return tuple(zip(self.params[carried:], self.args[carried:], strict=True))


__all__ = ["LoopRegion"]
