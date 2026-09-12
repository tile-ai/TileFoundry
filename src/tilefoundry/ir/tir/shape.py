"""``tir.ShapeOf`` — runtime shape value of a parameter's tensor at a given axis.

The expression's ``type`` is rank-0 ``TensorType`` of dtype ``i32`` -- a
scalar. An axis a tensor's type leaves open travels with its pointer, so the
extent this reads is already a parameter of the call; nothing derives one from
a name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.types.tensor_type import DType, TensorType


@dataclass(unsafe_hash=True)
class ShapeOf(Expr):
    """Runtime shape value of a parameter's tensor at a given axis."""

    type: TensorType = field(
        default_factory=lambda: TensorType.scalar(DType.i32), kw_only=True
    )
    param: Var
    axis: int


__all__ = ["ShapeOf"]
