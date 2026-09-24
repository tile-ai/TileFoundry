from __future__ import annotations

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
    "Swizzle",
    "ComposedLayout",
    "composition",
    "swizzle_of",
    "Topology",
    "check_topology",
    "append",
    "merge_mesh",
    "replace",
    "positions_below",
    "stated_layout",
    "level_axes",
    "level_index",
    "Placement",
    "covered_by_scope",
    "storage_reaches",
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
    "shard_layout_of",
]
