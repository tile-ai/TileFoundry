from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .dtype import BoolDType, DType, FloatDType, IntegerDType
from .tensor_type import TensorType, TupleType, Type, UnitType
from .utils import (
    bytes_by_storage,
    local_type_of,
    make_shard_tensor_type,
    make_tensor_type,
    numel,
    tensor_types,
    tensor_bytes,
    topology_extent,
)
from .callable_type import (
    CallableType,
    callable_type_for,
    callable_type_for_prim_function,
)


__all__ = [
    "BoolDType",
    "CallableType",
    "DType",
    "FloatDType",
    "IntegerDType",
    "TensorType",
    "TupleType",
    "Type",
    "UnitType",
    "callable_type_for",
    "callable_type_for_prim_function",
    "bytes_by_storage",
    "local_type_of",
    "make_shard_tensor_type",
    "make_tensor_type",
    "numel",
    "tensor_types",
    "tensor_bytes",
    "topology_extent",
]


def _register_dim_typeinfer() -> None:
    from . import dim, dim_typeinfer  # noqa: PLC0415, F401
