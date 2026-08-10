from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .int_tuple import IntTuple, flatten, product
from .layout import ComposedLayout, Layout, LayoutBase
from .layout_algebra import c_order_strides, prefix_product, try_c_order_strides
from .mesh import Mesh, Topology
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
)
from .utils import make_mesh

__all__ = [
    "IntTuple",
    "flatten",
    "product",
    "c_order_strides",
    "try_c_order_strides",
    "prefix_product",
    "LayoutBase",
    "Layout",
    "ComposedLayout",
    "Topology",
    "Mesh",
    "make_mesh",
    "ShardAttr",
    "Split",
    "Partial",
    "Broadcast",
    "Dynamic",
    "ShardLayout",
    "S",
    "P",
    "B",
    "canonical_shard_layout",
]
