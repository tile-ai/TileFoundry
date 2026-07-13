from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .dtype import LOW_PRECISION_DTYPES, DType, reject_low_precision
from .tensor_type import TensorType, TupleType, Type, UnitType
from .callable_type import (
    CallableType,
    callable_type_for,
    callable_type_for_prim_function,
)


__all__ = [
    "CallableType",
    "DType",
    "LOW_PRECISION_DTYPES",
    "TensorType",
    "TupleType",
    "Type",
    "UnitType",
    "callable_type_for",
    "callable_type_for_prim_function",
    "reject_low_precision",
]

# Lazy-register dim ops' typeinfer after core is fully loaded. We defer the
# side-effect import here because `dim` pulls `ir.core.expr.Expr` / `Op` and
# would create a cycle when `ir.core.expr` imports back `Type` from us.


def _register_dim_typeinfer() -> None:
    from . import dim, dim_typeinfer  # noqa: PLC0415, F401


# Triggered from tilefoundry.__init__ once both core and types.shard are loaded.
