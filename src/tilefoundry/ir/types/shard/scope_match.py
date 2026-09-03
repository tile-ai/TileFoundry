"""Match TIR scope declarations against required hardware participation.

The TIR verifier keeps its own tuple of enclosing mesh values. CUDA MMA uses
these structural predicates to decide whether an atom's required thread scope
is hosted by one of them; HIR execution-domain visibility is checked separately
by ``covered_by_scope`` and ``storage_reaches``.
"""

from __future__ import annotations

from ..storage import StorageKind, resolve_storage
from .int_tuple import product
from .layout import Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh, positions_at


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


def covered_by_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether *mesh* names no finer positions than the enclosing scope."""
    scope = {
        topology.name: positions_at(current, topology.name)
        for topology in current.topologies
    }
    return all(
        topology.name in scope
        and positions_at(mesh, topology.name) == scope[topology.name]
        for topology in mesh.topologies
    )


def storage_reaches(storage, mesh: Mesh, current: Mesh) -> bool:
    """Whether *storage* reaches across a coarser value-to-scope boundary."""
    if current.topologies[-1].name in {
        topology.name for topology in mesh.topologies
    }:
        return True
    try:
        storage = resolve_storage(storage)
    except (TypeError, ValueError):
        return False
    return storage in {StorageKind.GMEM, StorageKind.SMEM}


__all__ = [
    "covered_by_scope",
    "mesh_scope_matches_required_scope",
    "states_consistent_positions",
    "storage_reaches",
]
