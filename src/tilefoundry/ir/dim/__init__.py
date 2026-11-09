"""The behaviour of dimensions: arithmetic, isl conversion, normalization and substitution.

The layer above ``ir.types``. Importing it registers the ``simplify_dim``,
``normalize_dim`` and ``dim_range`` services that the layers below call.
See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from .isl_interop import (
    IslParamValues,
    dimension_to_isl_pw_aff,
    isl_to_dimension,
    layout_to_isl_map,
    shape_to_isl_set,
)
from .mesh_coord import MeshCoord, literal_axes
from .simplify import as_dimension, ceildiv, i64_const, is_dimension, maximum, minimum
from .substitute import (
    SubstitutionError,
    canonicalize_dims,
    canonicalize_topology_dims,
    collect_vars,
    has_symbolic_dims,
    substitute_dims,
    substitute_layout_dims,
    substitute_mesh_dims,
    substitute_shape_entry,
    substitute_topology_dims,
    var_names,
    vars_by_name,
)
