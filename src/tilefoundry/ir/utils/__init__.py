"""Helpers over types that need more than a type definition holds.

The layer above ``ir.dim`` and below ``ir.core``: it reads leaves, sizes, bounds and
local views of types, and nothing below imports it.
"""

from __future__ import annotations

from .bounds import (
    register_strides,
    shape_numel_upper_bound,
    shape_upper_bound,
    upper_bound,
)
from .compatible import (
    address_layout,
    canonical,
    canonical_layout,
    is_unsharded_layout,
    owner_annotation,
    types_compatible,
)
from .leaves import leaf_rank, leaf_span, leaves_of, map_tensor_types, result_fields, tensor_types
from .local import (
    divided_axes,
    filter_topologies,
    find_execution_mesh,
    local_type_of,
    make_shard_tensor_type,
    physical_layout,
    topology_level,
)
from .mesh_scope import (
    UnitChange,
    attrs_by_topology,
    axis_keys,
    covers_every_unit,
    equals_mesh_scope,
    first_unit_ids,
    flat_layout,
    from_meshes,
    runs,
    same_mesh,
    selected_units,
    storage_reaches,
    sub_box,
    topology_mesh,
    topology_names,
    unit_change,
    unit_ids,
    unit_layout,
    units_within,
    within_mesh_scope,
)
from .partials import (
    attrs_by_axis,
    check_multilinear_partials,
    reject_partials,
    require_matching_partial_state,
    require_uniform_partial_slices,
    varying_axis,
)
from .sizes import (
    bytes_by_storage,
    numel,
    shape_runtime_total,
    tensor_bytes,
)
from .tiler import axis_tiler
