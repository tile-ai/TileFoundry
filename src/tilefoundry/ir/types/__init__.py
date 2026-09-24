from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .dtype import BoolDType, DType, FloatDType, IntegerDType
from .tensor_type import TensorType, TupleType, Type, UnitType
# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .int_tuple import IntTuple, flatten, product
from .layout import ComposedLayout, Layout, LayoutBase, Swizzle
from .layout_algebra import (
    composition,
    swizzle_of,
)
from .mesh import (
    Mesh,
    Topology,
    append,
    check_topology,
    level_axes,
    level_index,
    make_mesh,
    merge_mesh,
    positions_below,
    replace,
    stated_layout,
)
from .placement import Placement
from .stride import c_order_strides, prefix_product, try_c_order_strides
from .scope_match import covered_by_scope, storage_reaches
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
    canonical_shard_layout,
    shard_layout_of,
)
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
    "B",
    "Broadcast",
    "ComposedLayout",
    "Dynamic",
    "IntTuple",
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
    "Swizzle",
    "Topology",
    "append",
    "c_order_strides",
    "canonical_shard_layout",
    "check_topology",
    "composition",
    "covered_by_scope",
    "flatten",
    "level_axes",
    "level_index",
    "make_mesh",
    "merge_mesh",
    "positions_below",
    "prefix_product",
    "product",
    "replace",
    "shard_layout_of",
    "stated_layout",
    "storage_reaches",
    "swizzle_of",
    "try_c_order_strides",
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
