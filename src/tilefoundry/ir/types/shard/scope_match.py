"""Match an enclosing mesh scope against an op's required thread scope."""
from __future__ import annotations

from .int_tuple import product
from .layout import Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh


def _as_layout(mesh: Mesh) -> Layout:
    return Layout(shape=tuple(mesh.layout.shape), strides=tuple(mesh.layout.strides))


def states_consistent_positions(mesh: Mesh) -> bool:
    declared = product(mesh.topologies)
    return declared is None or declared == size(mesh.layout)


def mesh_scope_matches_required_scope(current: Mesh, required: Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs."""
    # Same program topology level — a `cta` scope is never a `thread`/warp scope.
    if current.topologies[0].name != required.topologies[0].name:
        return False

    cur_domain = product(current.topologies)
    req_domain = product(required.topologies)
    if cur_domain is None or req_domain is None:
        return False

    cur_layout = _as_layout(current)
    req_layout = _as_layout(required)

    if not states_consistent_positions(current) or not states_consistent_positions(required):
        return False

    # Must be an admissible execution scope (injective, compact-ordered).
    if not is_inverse_projectable(cur_layout):
        return False

    # Same thread-value decomposition: the fragment's Split attrs index the mesh
    # axes, so the lane layout must match exactly (shape + strides) — a flat or
    # differently-shaped 32-lane scope cannot host a 2-axis (4, 8) fragment.
    return cur_layout.shape == req_layout.shape and cur_layout.strides == req_layout.strides


__all__ = ["mesh_scope_matches_required_scope", "states_consistent_positions"]
