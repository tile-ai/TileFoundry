"""``tir.ShapeOf`` — runtime shape value of a parameter's tensor at a given axis.

The expression's ``type`` is rank-0 ``TensorType`` of dtype ``i32`` — a
scalar. CUDA codegen lowers ``ShapeOf`` to the dedicated kernel scalar
parameter ``<param.name>_shape_<axis>`` introduced for the enclosing
``PrimFunction``; the host wrapper extracts that scalar from the
runtime tensor's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.types.tensor_type import DType, TensorType


@dataclass(frozen=True)
class ShapeOf(Expr):
    """Runtime shape value of a parameter's tensor at a given axis."""

    param: Var
    axis: int


def shape_var_name(param_name: str, axis: int) -> str:
    """Canonical kernel scalar parameter name for a ``ShapeOf(param, axis)``."""
    return f"{param_name}_shape_{axis}"


def parse_shape_var_name(name: str) -> tuple[str, int] | None:
    """Split ``<base>_shape_<axis>`` → ````, or ``None`` when *name* does not match the pattern.

    Split ``<base>_shape_<axis>`` → ``(base, axis)``, or ``None`` when *name*
    does not match the pattern. Tolerant: a user param that happens not to fit
    is not an error here — callers gate on :func:`is_hidden_shape_scalar`.
    """
    marker = "_shape_"
    idx = name.rfind(marker)
    if idx < 0:
        return None
    base = name[:idx]
    if not base:
        return None
    try:
        axis = int(name[idx + len(marker) :])
    except ValueError:
        return None
    return base, axis


def is_shape_scalar(param) -> bool:
    """A shape-scalar param is a rank-0 ``i32`` ``TensorType`` named ``<base>_shape_<axis>``.

    A shape-scalar param is a rank-0 ``i32`` ``TensorType`` named
    ``<base>_shape_<axis>`` — appended by lowering so a runtime tensor extent
    reaches the kernel. Not user-facing; the host wrapper derives it from the
    corresponding tensor arg's shape.
    """
    ty = param.type
    if not isinstance(ty, TensorType):
        return False
    if ty.shape:
        return False
    return ty.dtype is DType.i32


def is_hidden_shape_scalar(param, params) -> bool:
    """True iff *param* is a hidden shape scalar.

    True iff *param* is a hidden shape scalar: a rank-0 ``i32`` whose name
    parses as ``<base>_shape_<axis>`` AND ``<base>`` names a non-scalar tensor
    param in the same ``params`` with ``axis`` within that tensor's rank. A
    user-declared rank-0 ``i32`` (whose name does not reference another tensor
    param) is not hidden; neither is a malformed ``<base>_shape_<axis>`` whose
    axis is out of the base tensor's rank — the host wrapper could not derive
    such a scalar from a real shape axis.
    """
    if not is_shape_scalar(param):
        return False
    parsed = parse_shape_var_name(param.name)
    if parsed is None:
        return False
    base, axis = parsed
    for p in params:
        if p.name == base and isinstance(p.type, TensorType) and p.type.shape:
            return 0 <= axis < len(p.type.shape)
    return False


__all__ = [
    "ShapeOf",
    "shape_var_name",
    "parse_shape_var_name",
    "is_shape_scalar",
    "is_hidden_shape_scalar",
]
