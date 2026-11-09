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
from .compatible import is_unsharded_layout, types_compatible
from .leaves import leaf_rank, leaf_span, leaves_of, tensor_types
from .local import (
    divided_mesh_axes,
    local_type_of,
    make_shard_tensor_type,
    participant_frame,
    participant_layout,
    physical_layout,
    try_local_type_of,
)
from .mesh_scope import (
    covered_by_scope,
    device_layout,
    mesh_scope_matches_required_scope,
    states_consistent_positions,
    storage_reaches,
)
from .partials import (
    attrs_by_axis,
    check_multilinear_partials,
    reject_partials,
    require_matching_partial_state,
    require_uniform_partial_slices,
)
from .sizes import (
    bytes_by_storage,
    numel,
    shape_runtime_total,
    tensor_bytes,
    topology_extent,
)
from .tiler import axis_tiler
