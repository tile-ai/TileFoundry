"""Which scope a program runs in, and what that scope admits.

Entering a `with Mesh(...)` composes the scope in force with the one it names;
a value is read where the positions it is laid out over are the positions the
scope runs on. Both questions are about the mesh a statement stands inside,
which is neither a type nor a visitor, so they live here rather than with
either.
"""

from __future__ import annotations

from tilefoundry import services
from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple, layout_algebra


def device_layout(mesh: types.Mesh) -> types.Layout:
    """The whole mesh's axes as the device numbers its positions.

    Each level states its own numbering, and a position of one level is a
    position within its parent, so an axis steps by what it states times what
    the levels under it hold. That product is the numbering a coordinate of
    the whole mesh is read in, and the one a mesh is written down in.
    """
    shape: list = []
    strides: list = []
    sizes = tuple(getattr(topology, "size", 1) for topology in mesh.topologies)
    units = types.compact_major(sizes, major="row") if all(
        isinstance(one, int) for one in sizes
    ) else (1,) * len(sizes)
    for arrangement, unit in zip(types.levels(mesh), units):
        stated = arrangement.strides
        shape.extend(types.flatten(arrangement.shape))
        strides.extend(
            step * unit
            for step in (
                types.flatten(stated)
                if stated is not None
                else types.compact_major(tuple(types.flatten(arrangement.shape)))
            )
        )
    return types.Layout(shape=tuple(shape), strides=tuple(strides))


def level_selection(arrangement, start):
    """One level of a scope as the positions it selects: a size when they run contiguously."""
    reduced = layout_algebra.filter(arrangement, major="row")
    if layout_algebra.is_contiguous(reduced, major="row"):
        return services.normalize_dim(types.size(reduced)), start
    return reduced, start


def covered_by_scope(mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether *mesh* selects exactly the positions the enclosing scope does.

    Level by level, on the positions each level states rather than on the axes
    standing where it does: a scope that is part of a level -- one warp of a
    CTA's threads -- says which positions it is by where its run starts and how
    its modes step, and a value laid out over that same run is inside it
    however either of them wrote the axes down.
    """
    scope = {
        getattr(topology, "name", topology): level_selection(arrangement, start)
        for topology, arrangement, start in zip(
            current.topologies, types.levels(current), types.starts(current)
        )
    }
    return all(
        getattr(topology, "name", topology) in scope
        and level_selection(arrangement, start)
        == scope[getattr(topology, "name", topology)]
        for topology, arrangement, start in zip(
            mesh.topologies, types.levels(mesh), types.starts(mesh)
        )
    )


def storage_reaches(storage, mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether *storage* reaches across a coarser value-to-scope boundary."""
    if current.topologies[-1].name in {topology.name for topology in mesh.topologies}:
        return True
    try:
        storage = expr.resolve_storage(storage)
    except (TypeError, ValueError):
        return False
    return storage in {expr.StorageKind.GMEM, expr.StorageKind.SMEM}


def states_consistent_positions(mesh: types.Mesh) -> bool:
    """Whether the positions a mesh states are the ones its levels declare."""
    return int_tuple.product(mesh.topologies) == types.size(mesh.layout)


def mesh_scope_matches_required_scope(current: types.Mesh, required: types.Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs."""
    if current.topologies[0].name != required.topologies[0].name:
        return False
    if not states_consistent_positions(current) or not states_consistent_positions(required):
        return False
    here, there = flat_layout(current), flat_layout(required)
    if not layout_algebra.is_inverse_projectable(here):
        return False
    return here.shape == there.shape and here.strides == there.strides


def flat_layout(mesh: types.Mesh) -> types.Layout:
    stated = mesh.layout.outer if isinstance(mesh.layout, types.ComposedLayout) else mesh.layout
    strides = stated.strides
    return types.Layout(
        shape=tuple(types.flatten(stated.shape)),
        strides=tuple(types.flatten(strides)),
    )


__all__ = [
    "device_layout",
    "covered_by_scope",
    "mesh_scope_matches_required_scope",
    "states_consistent_positions",
    "storage_reaches",
]
