"""Type, layout and mesh definitions.

The layer above ``ir.expr``: it reads dimensions through ``ir.expr.dim`` and the
``services`` and never imports ``ir.dim``. ``int_tuple`` and ``layout_algebra`` are
exported as namespaces because their function names (``product``, ``filter``,
``slice``) would collide. ``get`` is the one of ``mesh``: it also reads a mesh.
``domain_offset`` is the one of ``shard_layout``.
See [types](docs/spec/types.md) and [shard](docs/spec/shard.md).
"""

from __future__ import annotations

from . import int_tuple, layout_algebra
from .callable_type import (
    CallableType,
    bind_annotation,
    callable_type_for,
    callable_type_for_prim_function,
)
from .int_tuple import IntTuple, flatten_to_tuple
from .layout import (
    ComposedLayout,
    Layout,
    LayoutBase,
    NotProjectableError,
    Swizzle,
    coshape,
    cosize,
    depth,
    flatten,
    plain_layout,
    rank,
    select,
    size,
    take,
    unflatten,
)
from .layout_algebra import (
    make_composed_layout,
    make_layout,
    make_layout_like,
    make_ordered_layout,
)
from .mesh import (
    Mesh,
    axis_keys,
    check_topology,
    equals_mesh_scope,
    flat_layouts,
    get,
    layouts,
    offset,
    selected_units,
    sub_box,
    sub_meshes,
    topology_names,
    within_mesh_scope,
)
from .placement import Placement
from .pointer import PointerType
from .shard_layout import (
    B,
    Broadcast,
    P,
    Partial,
    S,
    ShardAttr,
    ShardLayout,
    Split,
    V,
    Varying,
    combine_attrs,
    domain_offset,
    local_layout,
    local_layout_and_offset,
    local_readback,
    mesh_coords,
    normalize_layout,
    partial_reductions_by_axis,
    physical_attr,
    shard_layout_local_shape,
    shard_layout_of,
    split_target_axes,
)
from .stride import (
    LayoutLeft,
    LayoutRight,
    compact_col_major,
    compact_major,
    compact_order,
    compact_row_major,
    crd2crd,
    crd2idx,
    idx2crd,
)
from .swizzle_layout import get_nonswizzle_portion, get_swizzle_portion, make_swizzle
from .tensor_type import ShapeDim, TensorType, TupleType, Type
from .topology import Topology, topology_name
from .visitor import TypeFunctor, TypeMutator, TypeVisitor
