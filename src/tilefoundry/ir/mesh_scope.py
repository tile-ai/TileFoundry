"""Which scope a program runs in, and what that scope admits.

Entering a `with Mesh(...)` composes the scope in force with the one it names;
a value is read where the positions it is laid out over are the positions the
scope runs on. Both questions are about the mesh a statement stands inside,
which is neither a type nor a visitor, so they live here rather than with
either.
"""

from __future__ import annotations

from tilefoundry.ir.types.int_tuple import flatten, product
from tilefoundry.ir.types.layout import ComposedLayout, Layout, get, rank, size
from tilefoundry.ir.types.layout_algebra import is_inverse_projectable
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.ir.types.storage import StorageKind, resolve_storage
from tilefoundry.ir.types.stride import compact_major, idx2crd


def _levels(mesh: Mesh) -> tuple[Layout, ...]:
    """Each level's own arrangement: mode ``i`` of the mesh is level ``i``."""
    stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
    if stated is None:
        raise ValueError(
            "a mesh whose slice states an identity box states no arrangement of "
            "its own, so its levels select nothing"
        )
    return tuple(get(stated, index) for index in range(rank(stated)))


def _starts(mesh: Mesh) -> tuple[int, ...]:
    """Where each level's run begins, read out of the offset the mesh states.

    The offset is one index in the numbering the device gives every position,
    and the levels are its shape, so the coordinate it stands for is what each
    level starts at.
    """
    offset = mesh.layout.offset if isinstance(mesh.layout, ComposedLayout) else 0
    sizes = tuple(getattr(topology, "size", 1) for topology in mesh.topologies)
    if not isinstance(offset, int) or any(not isinstance(one, int) for one in sizes):
        return (0,) * len(sizes)
    return tuple(idx2crd(offset, sizes, compact_major(sizes)))


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
    for arrangement, unit in zip(_levels(mesh), units):
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
            current.topologies, _levels(current), _starts(current)
        )
    }
    return all(
        getattr(topology, "name", topology) in scope
        and _selected(arrangement, start)
        == scope[getattr(topology, "name", topology)]
        for topology, arrangement, start in zip(
            mesh.topologies, _levels(mesh), _starts(mesh)
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


def check_topology(mesh: Mesh) -> None:
    """Reject static mesh positions beyond their declared topology extents.

    A constant slice is already bounded by ``Mesh.__getitem__``; its shortened
    axes no longer land on full topology boundaries and are therefore accepted.
    """
    if isinstance(mesh.layout, ComposedLayout):
        return
    for topology, arrangement in zip(mesh.topologies, _levels(mesh)):
        declared = getattr(topology, "size", None)
        if not isinstance(declared, int) or isinstance(declared, bool):
            continue
        count = product(tuple(flatten(arrangement.shape)))
        if isinstance(count, int) and count > declared:
            raise ValueError(
                f"mesh level {getattr(topology, 'name', topology)!r} has {count} "
                f"positions, exceeding declared extent {declared}"
            )


def _joined(topologies: tuple, levels: tuple, names: tuple) -> Mesh:
    """One mesh out of the levels it names, each stating its own arrangement."""
    if len(levels) == 1:
        return Mesh(topologies=topologies, layout=levels[0], names=names)
    return Mesh(
        topologies=topologies,
        layout=Layout(
            shape=tuple(tuple(flatten(one.shape)) for one in levels),
            strides=tuple(tuple(flatten(one.strides)) for one in levels),
        ),
        names=names,
    )


def _named(mesh: Mesh) -> tuple[str, ...]:
    return tuple(getattr(topology, "name", topology) for topology in mesh.topologies)


def merge_mesh(meshes: "tuple[Mesh, ...]") -> Mesh:
    """The scope in force once each of *meshes* has been entered in turn.

    A scope naming levels none of those in force name is appended below them.
    One naming every level in force replaces them. One naming a suffix of them
    replaces that suffix and keeps what is above. Any other overlap is refused
    rather than decomposed: which positions the half-named levels would then
    state is nobody's statement. No stride is rescaled, because every level
    already states its own numbering.
    """
    result = meshes[0]
    for inner in meshes[1:]:
        here, there = _named(result), _named(inner)
        if set(here).isdisjoint(there):
            result = _joined(
                (*result.topologies, *inner.topologies),
                (*_levels(result), *_levels(inner)),
                (*result.names, *inner.names),
            )
        elif set(here) <= set(there):
            result = inner
        elif len(there) < len(here) and here[-len(there) :] == there:
            kept = len(here) - len(there)
            above = _levels(result)[:kept]
            named = sum(len(flatten(one.shape)) for one in above)
            result = _joined(
                (*result.topologies[:kept], *inner.topologies),
                (*above, *_levels(inner)),
                (*result.names[:named], *inner.names),
            )
        else:
            shared = sorted(set(here) & set(there))
            unnamed = sorted(set(here) - set(there))
            raise ValueError(
                f"{shared} named again while {unnamed} is not; a scope either "
                "replaces the levels in force or adds levels below them"
            )
    check_topology(result)
    return result


__all__ = [
    "check_topology",
    "device_layout",
    "covered_by_scope",
    "merge_mesh",
    "mesh_scope_matches_required_scope",
    "states_consistent_positions",
    "storage_reaches",
]
