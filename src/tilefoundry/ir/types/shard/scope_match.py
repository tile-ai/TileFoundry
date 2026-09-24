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
from .mesh import Mesh, level_positions, positions_at


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


def _stated_positions(mesh: Mesh, topology_level: str) -> tuple[tuple, tuple]:
    """One level's positions as written, with the axes of one position left out.

    An axis of one position names no instance, so two scopes state the same
    positions whether or not either of them wrote such an axis down.
    """
    shape, strides = positions_at(mesh, topology_level)
    kept = tuple(axis for axis, extent in enumerate(shape) if extent != 1)
    return tuple(shape[axis] for axis in kept), tuple(strides[axis] for axis in kept)


def covered_by_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether *mesh* selects exactly the positions the enclosing scope does.

    Level by level, and on the positions each level states rather than on the
    axes standing where it does: a scope that is part of a level -- one warp of
    a CTA's threads -- says which positions it is by where its run starts and
    how its modes step. Where neither states those as numbers, a grid sized by
    a dimension nobody has fixed yet, they are compared as written instead: the
    axes each level was given, which is the only answer there is then.
    """
    mine, scope = level_positions(mesh), level_positions(current)
    if mine is not None and scope is not None:
        return all(name in scope and positions == scope[name] for name, positions in mine.items())
    written = {
        topology.name: _stated_positions(current, topology.name)
        for topology in current.topologies
    }
    return all(
        topology.name in written
        and _stated_positions(mesh, topology.name) == written[topology.name]
        for topology in mesh.topologies
    )


def storage_reaches(storage, mesh: Mesh, current: Mesh) -> bool:
    """Whether *storage* reaches across a coarser value-to-scope boundary."""
    if current.topologies[-1].name in {topology.name for topology in mesh.topologies}:
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
