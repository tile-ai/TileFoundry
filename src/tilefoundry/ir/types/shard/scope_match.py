"""Match an enclosing mesh scope against an op's required thread scope."""

from __future__ import annotations

from .int_tuple import product
from .layout import Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh


def _as_layout(mesh: Mesh) -> Layout:
    return Layout(shape=tuple(mesh.layout.shape), strides=tuple(mesh.layout.strides))


def states_consistent_positions(mesh: Mesh) -> bool:
    return product(mesh.topologies) == size(mesh.layout)


def mesh_scope_matches_required_scope(current: Mesh, required: Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs."""
    if current.topologies[0].name != required.topologies[0].name:
        return False

    cur_layout = _as_layout(current)
    req_layout = _as_layout(required)

    if not states_consistent_positions(current) or not states_consistent_positions(required):
        return False

    if not is_inverse_projectable(cur_layout):
        return False

    return cur_layout.shape == req_layout.shape and cur_layout.strides == req_layout.strides


__all__ = ["mesh_scope_matches_required_scope", "states_consistent_positions"]
