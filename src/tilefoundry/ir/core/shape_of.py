"""``tir.ShapeOf`` — runtime shape value of a parameter's tensor at a given axis.

The expression's ``type`` is rank-0 ``TensorType`` of dtype ``i32`` -- a
scalar. An axis a tensor's type leaves open travels with its pointer, so the
extent this reads is already a parameter of the call; nothing derives one from
a name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir import expr, types


@dataclass(unsafe_hash=True)
class ShapeOf(expr.Expr):
    """Runtime shape value of a parameter's tensor at a given axis."""

    type: types.TensorType = field(
        default_factory=lambda: types.TensorType.from_scalar(expr.DType.i32), kw_only=True
    )
    param: expr.Var
    axis: int


def shape_of(param: expr.Var, *, axis: int) -> ShapeOf:
    """Author the same runtime extent expression that the TIR printer emits."""
    return ShapeOf(param=param, axis=axis)


__all__ = ["ShapeOf", "shape_of"]
