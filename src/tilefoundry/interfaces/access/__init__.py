"""Access relations and buffer aliases registered per Op."""

from . import mesh_coord
from .alias import alias_registry, aliased_operand, register_alias
from .builders import (
    broadcast,
    broadcast_all,
    broadcast_shapes,
    gather_relations,
    identity,
    identity_relations,
    linearized_view,
    matmul_relations,
    readnone_relations,
    state,
    unread,
)
from .iteration import iterating, iteration_geometry
from .loop_domain import loop_domain
from .physical import byte_relation
from .projection import (
    affine_term,
    factored_image,
    local_relations_of,
    logical_axes_of,
    logical_coordinates,
    logical_term,
    projected,
    projected_axes,
    renaming_relation,
    shape_from_relation,
)
from .reach import (
    leaf_paths,
    reached_elements,
    reached_leaves,
    settled,
    static_bytes,
)
from .registry import (
    iteration_universe,
    register_relation,
    relation_registry,
    relations_of,
)
from .relation import Relation, affine_of, param_name
from .window import (
    control_read,
    normalised_rows,
    placed_window,
    reached_at,
    window_source,
)
