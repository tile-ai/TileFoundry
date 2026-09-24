"""Match TIR scope declarations against required hardware participation.

The TIR verifier keeps its own tuple of enclosing mesh values. CUDA MMA uses
these structural predicates to decide whether an atom's required thread scope
is hosted by one of them; HIR execution-domain visibility is checked separately
by ``covered_by_scope`` and ``storage_reaches``.
"""

from __future__ import annotations

from ..storage import StorageKind, resolve_storage
from .int_tuple import flatten, product
from .layout import Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh, stated_layout


def _as_layout(mesh: Mesh) -> Layout:
    return mesh.positions


def states_consistent_positions(mesh: Mesh) -> bool:
    return product(mesh.topologies) == size(mesh.positions)


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


def _selected(arrangement) -> tuple[tuple, tuple, int]:
    """One level's positions as a set of them reads: its modes, and where it starts.

    An axis of one position names no instance, and modes written in another
    order state the same positions, so the modes come back sorted by step with
    the adjacent ones joined and the ones of a single position left out.
    """
    offset = arrangement.offset if hasattr(arrangement, "offset") else 0
    stated = stated_layout(arrangement)
    modes = [
        (extent, stride)
        for extent, stride in zip(flatten(stated.shape), flatten(stated.strides))
        if extent != 1
    ]
    joined: list[list] = []
    for extent, stride in sorted(modes, key=lambda mode: (mode[1], mode[0])):
        if joined and joined[-1][0] * joined[-1][1] == stride:
            joined[-1][0] *= extent
        else:
            joined.append([extent, stride])
    return (
        tuple(extent for extent, _ in joined),
        tuple(stride for _, stride in joined),
        offset,
    )


def covered_by_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether *mesh* selects exactly the positions the enclosing scope does.

    Level by level, on the positions each level states rather than on the axes
    standing where it does: a scope that is part of a level -- one warp of a
    CTA's threads -- says which positions it is by where its run starts and how
    its modes step, and a value laid out over that same run is inside it
    however either of them wrote the axes down.
    """
    scope = {
        getattr(topology, "name", topology): _selected(arrangement)
        for topology, arrangement in zip(current.topologies, current.levels)
    }
    return all(
        getattr(topology, "name", topology) in scope
        and _selected(arrangement) == scope[getattr(topology, "name", topology)]
        for topology, arrangement in zip(mesh.topologies, mesh.levels)
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
