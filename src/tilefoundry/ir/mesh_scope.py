"""Which scope a program runs in, and what that scope admits.

Entering a `with Mesh(...)` composes the scope in force with the one it names;
a value is read where the positions it is laid out over are the positions the
scope runs on. Both questions are about the mesh a statement stands inside,
which is neither a type nor a visitor, so they live here rather than with
either.
"""

from __future__ import annotations

from tilefoundry.ir.types.int_tuple import flatten, product
from tilefoundry.ir.types.layout import ComposedLayout, Layout, size
from tilefoundry.ir.types.layout_algebra import is_inverse_projectable
from tilefoundry.ir.types.mesh import Mesh, check_topology, levels, starts
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.ir.types.stride import compact_major


def device_layout(mesh: Mesh) -> Layout:
    """The whole mesh's axes as the device numbers its positions.

    Each level states its own numbering, and a position of one level is a
    position within its parent, so an axis steps by what it states times what
    the levels under it hold. That product is the numbering a coordinate of
    the whole mesh is read in, and the one a mesh is written down in.
    """
    shape: list = []
    strides: list = []
    sizes = tuple(getattr(topology, "size", 1) for topology in mesh.topologies)
    units = compact_major(sizes, major="row") if all(
        isinstance(one, int) for one in sizes
    ) else (1,) * len(sizes)
    for arrangement, unit in zip(levels(mesh), units):
        stated = arrangement.strides
        shape.extend(flatten(arrangement.shape))
        strides.extend(
            step * unit
            for step in (
                flatten(stated)
                if stated is not None
                else compact_major(tuple(flatten(arrangement.shape)))
            )
        )
    return Layout(shape=tuple(shape), strides=tuple(strides))


def _selected(arrangement: Layout, start: int) -> tuple[tuple, tuple, int]:
    """One level's positions as a set of them reads: its modes, and where it starts.

    An axis of one position names no instance, and modes written in another
    order state the same positions, so the modes come back sorted by step with
    the adjacent ones joined and the ones of a single position left out.
    """
    strides = arrangement.strides
    if strides is None:
        return tuple(flatten(arrangement.shape)), (), start
    modes = [
        (extent, stride)
        for extent, stride in zip(flatten(arrangement.shape), flatten(strides))
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
        start,
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
        getattr(topology, "name", topology): _selected(arrangement, start)
        for topology, arrangement, start in zip(
            current.topologies, levels(current), starts(current)
        )
    }
    return all(
        getattr(topology, "name", topology) in scope
        and _selected(arrangement, start)
        == scope[getattr(topology, "name", topology)]
        for topology, arrangement, start in zip(
            mesh.topologies, levels(mesh), starts(mesh)
        )
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


def states_consistent_positions(mesh: Mesh) -> bool:
    """Whether the positions a mesh states are the ones its levels declare."""
    return product(mesh.topologies) == size(mesh.layout)


def mesh_scope_matches_required_scope(current: Mesh, required: Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs."""
    if current.topologies[0].name != required.topologies[0].name:
        return False
    if not states_consistent_positions(current) or not states_consistent_positions(required):
        return False
    here, there = _flat(current), _flat(required)
    if not is_inverse_projectable(here):
        return False
    return here.shape == there.shape and here.strides == there.strides


def _flat(mesh: Mesh) -> Layout:
    stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
    strides = stated.strides
    return Layout(
        shape=tuple(flatten(stated.shape)),
        strides=None if strides is None else tuple(flatten(strides)),
    )


__all__ = [
    "check_topology",
    "device_layout",
    "covered_by_scope",
    "mesh_scope_matches_required_scope",
    "states_consistent_positions",
    "storage_reaches",
]
