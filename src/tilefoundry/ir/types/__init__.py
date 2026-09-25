from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .dtype import BoolDType, DType, FloatDType, IntegerDType
from .storage import StorageKind
from .int_tuple import IntTuple
from .layout import ComposedLayout, Layout, LayoutBase, Swizzle
from .shard_layout import (
    B,
    Broadcast,
    Dynamic,
    P,
    Partial,
    S,
    ShardAttr,
    ShardLayout,
    Split,
)
from .mesh import Mesh, Topology, make_mesh
from .placement import Placement
from .tensor_type import TensorType, TupleType, Type, UnitType
from .utils import make_shard_tensor_type, make_tensor_type
from .callable_type import CallableType, callable_type_for


__all__ = [
    "B",
    "BoolDType",
    "Broadcast",
    "CallableType",
    "ComposedLayout",
    "DType",
    "Dynamic",
    "FloatDType",
    "IntTuple",
    "IntegerDType",
    "Layout",
    "LayoutBase",
    "Mesh",
    "P",
    "Partial",
    "Placement",
    "S",
    "ShardAttr",
    "ShardLayout",
    "Split",
    "StorageKind",
    "Swizzle",
    "TensorType",
    "Topology",
    "TupleType",
    "Type",
    "UnitType",
    "callable_type_for",
    "make_mesh",
    "make_shard_tensor_type",
    "make_tensor_type",
]


def _register_dim_typeinfer() -> None:
    from . import dim, dim_typeinfer  # noqa: PLC0415, F401
