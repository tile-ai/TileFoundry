from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.int_tuple import product
from tilefoundry.ir.types.layout import ComposedLayout, Layout, LayoutBase, flatten, get
from tilefoundry.ir.types.layout import rank as _rank
from tilefoundry.ir.types.stride import compact_major, compact_row_major, crd2idx, idx2crd
from tilefoundry.ir.types.tensor_type import ShapeDim


@dataclass(frozen=True)
class Topology:
    """Name one hardware level and its explicit static or symbolic size."""

    name: str

    size: "ShapeDim"

    def __post_init__(self) -> None:
        if self.size is None:
            raise ValueError(
                f"Topology {self.name!r}: extent must be explicit; None is not "
                "a ShapeDim. The rule: "
                "tilefoundry spec target topology-levels"
            )


@dataclass(frozen=True)
class Mesh:
    """Describe hardware levels, logical positions, and axis names.

    ``layout`` runs parallel to ``topologies``: one arrangement per level, in
    that level's own numbering, so a level states which of its positions it
    selects without any level above it entering the answer. A constant slice
    replaces one level's arrangement with a ``ComposedLayout`` whose ``offset``
    is where that level's run starts.

    See [shard §5](docs/spec/shard.md#5-mesh).
    """

    topologies: tuple[Topology | str, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.topologies) > 1 and any(
            not isinstance(topology, Topology) for topology in self.topologies
        ):
            raise ValueError("a multi-level Mesh requires Topology values")
        topology_names = tuple(
            getattr(topology, "name", topology) for topology in self.topologies
        )
        if len(set(topology_names)) != len(topology_names):
            raise ValueError(f"Mesh topology names must be unique, got {topology_names!r}")
        object.__setattr__(self, "layout", _nested(self.layout, tuple(self.topologies)))
        for axis, extent in enumerate(flatten(self.layout.shape)):
            if extent is None:
                raise ValueError(
                    f"Mesh: layout axis {axis} must have an explicit extent; "
                    "None is not a ShapeDim. The rule: tilefoundry spec shard mesh"
                )

    def __getitem__(self, key) -> "Mesh":
        """Return a constant sub-mesh selected by integers or unit-step slices.

        Missing axes are full slices; integers select extent one. The result
        preserves topology and names while recording the sub-box as a
        ``ComposedLayout``. Each level retains its own arrangement, while the
        slice offset uses the device's numbering across all levels.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        held_levels = levels(self)
        rank = sum(len(flatten(level.shape)) for level in held_levels)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > rank:
            raise ValueError(f"mesh slice has {len(keys)} indices but the mesh has {rank} axes")
        keys = keys + (slice(None),) * (rank - len(keys))

        sub_levels: list[Layout] = []
        offset = 0
        axis = 0
        units = (
            (1,)
            if len(self.topologies) == 1
            else compact_major(tuple(topology.size for topology in self.topologies))
        )
        for level, unit in zip(held_levels, units):
            level_shape = tuple(flatten(level.shape))
            stated = level.strides
            level_strides = (
                tuple(flatten(stated)) if stated is not None else compact_row_major(level_shape)
            )
            sub_shape: list[int] = []
            for k, extent, stride in zip(
                keys[axis : axis + len(level_shape)], level_shape, level_strides
            ):
                if not isinstance(extent, int) or not isinstance(stride, int):
                    raise ValueError(f"cannot slice mesh axis {axis} with a dynamic extent/stride")
                if isinstance(k, int):
                    start = k + extent if k < 0 else k
                    if not (0 <= start < extent):
                        raise ValueError(
                            f"mesh slice index {k} out of range for axis {axis} (extent {extent})"
                        )
                    selected = 1
                elif isinstance(k, slice):
                    if k.step not in (None, 1):
                        raise ValueError(f"mesh slice step must be 1 (axis {axis})")
                    start = 0 if k.start is None else (k.start + extent if k.start < 0 else k.start)
                    stop = extent if k.stop is None else (k.stop + extent if k.stop < 0 else k.stop)
                    if not (0 <= start <= stop <= extent):
                        raise ValueError(
                            f"mesh slice {k.start}:{k.stop} out of range for axis "
                            f"{axis} (extent {extent})"
                        )
                    selected = stop - start
                    if selected == 0:
                        raise ValueError(f"mesh slice selects an empty range on axis {axis}")
                else:
                    raise ValueError(
                        f"mesh slice index must be int or slice, got {type(k).__name__}"
                    )
                offset += start * stride * unit
                sub_shape.append(selected)
                axis += 1
            sub_levels.append(Layout(tuple(sub_shape), level_strides))

        return Mesh(
            topologies=self.topologies,
            layout=ComposedLayout(
                inner=None,
                offset=offset,
                outer=_joined_layout(tuple(sub_levels)),
            ),
            names=self.names,
        )


def _levelled(layout, topologies: tuple) -> bool:
    """Whether *layout* already states one mode per level the mesh names."""
    return _rank(layout) == len(topologies) and all(
        isinstance(mode, tuple) for mode in layout.shape
    )


def _nested(layout, topologies: tuple) -> "Layout | ComposedLayout":
    """What a mesh was written as, with every level's axes under its own mode.

    A mesh naming one level states that level's arrangement directly. One
    naming several may state a tuple of extents or one arrangement over all of
    their axes; the axes are handed to the levels left to right, each taking
    them until their extents multiply to its own size, and each level's steps
    are divided by what the levels under it hold. So mode ``i`` of what comes
    back is level ``i``'s own arrangement, and a boundary no prefix of axes
    lands on is refused rather than guessed.
    """
    if not isinstance(layout, LayoutBase):
        extents = tuple(flatten(layout))
        layout = Layout(shape=extents, strides=compact_row_major(extents))
    if isinstance(layout, ComposedLayout):
        if layout.outer is None or _levelled(layout.outer, topologies):
            return layout
        return ComposedLayout(
            inner=layout.inner,
            offset=layout.offset,
            outer=_nested(layout.outer, topologies),
        )
    if _levelled(layout, topologies):
        return layout

    extents = tuple(flatten(layout.shape))
    stated = layout.strides if layout.strides is not None else compact_row_major(extents)
    steps = tuple(flatten(stated))
    if len(topologies) == 1:
        return Layout(shape=(extents,), strides=(steps,))
    units: list[int] = []
    below = 1
    for topology in reversed(topologies):
        units.insert(0, below)
        size = topology.size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(
                f"mesh level {topology.name!r} states extent "
                f"{size!r}; cutting one arrangement at the level boundaries needs "
                "each of their position counts"
            )
        below *= size

    shape: list = []
    strides: list = []
    axis = 0
    for topology, unit in zip(topologies, units):
        size = topology.size
        taken_extents: list = []
        taken_steps: list = []
        reach = 1
        while axis < len(extents) and (reach < size or extents[axis] == 1):
            extent, step = extents[axis], steps[axis]
            if not isinstance(extent, int) or isinstance(extent, bool):
                raise ValueError(
                    f"mesh layout axis {axis} states extent {extent!r}; cutting one "
                    "arrangement at the level boundaries needs concrete axis extents"
                )
            if not isinstance(step, int) or isinstance(step, bool) or step % unit:
                raise ValueError(
                    f"mesh axis {axis} steps by {step!r}, which the {unit} positions "
                    f"below {topology.name!r} do not divide; its positions are not "
                    "that level's"
                )
            reach *= extent
            taken_extents.append(extent)
            taken_steps.append(step // unit)
            axis += 1
        if reach != size:
            raise ValueError(
                f"mesh axes {extents} do not land on the boundary of level "
                f"{topology.name!r} at {size}: the axes up to there multiply to "
                f"{reach}. Write the axis that straddles it as the two axes it is"
            )
        shape.append(tuple(taken_extents))
        strides.append(tuple(taken_steps))
    if axis != len(extents):
        raise ValueError(
            f"mesh layout has {len(extents)} axes but the levels it names account "
            f"for {axis}; every axis belongs to one of them"
        )
    return Layout(shape=tuple(shape), strides=tuple(strides))


def levels(mesh: Mesh) -> tuple[Layout, ...]:
    """Each level's arrangement, in that level's own numbering."""
    stated = mesh.layout.outer if isinstance(mesh.layout, ComposedLayout) else mesh.layout
    if stated is None:
        raise ValueError(
            "a mesh whose slice states an identity box states no arrangement of "
            "its own, so its levels select nothing"
        )
    return tuple(get(stated, index) for index in range(_rank(stated)))


def starts(mesh: Mesh) -> tuple[int, ...]:
    """Where each level's run starts, decoded from the device-numbered offset."""
    offset = mesh.layout.offset if isinstance(mesh.layout, ComposedLayout) else 0
    if not isinstance(offset, int):
        return (0,) * len(mesh.topologies)
    if len(mesh.topologies) == 1:
        return (offset,)
    sizes = tuple(topology.size for topology in mesh.topologies)
    if any(not isinstance(one, int) for one in sizes):
        return (0,) * len(sizes)
    return tuple(idx2crd(offset, sizes, compact_major(sizes)))


def selected_run(arrangement: Layout, start: int) -> tuple[tuple, tuple, int]:
    """Reduce one level's selected positions to its joined modes and start."""
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


def _continuous_interval(run: tuple[tuple, tuple, int]) -> tuple[int, int] | None:
    extents, strides, start = run
    if not isinstance(start, int):
        return None
    if not extents:
        return start, start + 1
    if len(extents) != 1 or strides != (1,) or not isinstance(extents[0], int):
        return None
    return start, start + extents[0]


def within_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether each continuous run selected by *mesh* is within *current*."""
    scope = {
        getattr(topology, "name", topology): selected_run(arrangement, start)
        for topology, arrangement, start in zip(
            current.topologies, levels(current), starts(current)
        )
    }
    for topology, arrangement, start in zip(
        mesh.topologies, levels(mesh), starts(mesh)
    ):
        name = getattr(topology, "name", topology)
        inner = _continuous_interval(selected_run(arrangement, start))
        outer = _continuous_interval(scope[name]) if name in scope else None
        if inner is None or outer is None:
            return False
        if not (outer[0] <= inner[0] and inner[1] <= outer[1]):
            return False
    return True


def check_topology(mesh: Mesh) -> None:
    """Reject static mesh positions beyond their declared topology extents.

    A constant slice is already bounded by ``Mesh.__getitem__``; its shortened
    axes no longer land on full topology boundaries and are therefore accepted.
    """
    if isinstance(mesh.layout, ComposedLayout):
        return
    for topology, arrangement in zip(mesh.topologies, levels(mesh)):
        declared = getattr(topology, "size", None)
        if not isinstance(declared, int) or isinstance(declared, bool):
            continue
        count = product(tuple(flatten(arrangement.shape)))
        if isinstance(count, int) and count > declared:
            raise ValueError(
                f"mesh level {getattr(topology, 'name', topology)!r} has {count} "
                f"positions, exceeding declared extent {declared}"
            )


def _joined_layout(levels: tuple[Layout, ...]) -> Layout:
    if len(levels) == 1:
        return levels[0]
    return Layout(
        shape=tuple(tuple(flatten(level.shape)) for level in levels),
        strides=tuple(tuple(flatten(level.strides)) for level in levels),
    )


def _joined(
    topologies: tuple[Topology, ...],
    levels: tuple[Layout, ...],
    starts: tuple[int, ...],
    names: tuple[str, ...],
    *,
    sliced: bool,
) -> Mesh:
    layout: Layout | ComposedLayout = _joined_layout(levels)
    if sliced:
        sizes = tuple(topology.size for topology in topologies)
        if not all(isinstance(size, int) for size in sizes):
            raise ValueError("joining sliced meshes needs static topology extents")
        layout = ComposedLayout(None, crd2idx(starts, sizes, compact_major(sizes)), layout)
    return Mesh(topologies, layout, names)


def _named(mesh: Mesh) -> tuple[str, ...]:
    return tuple(getattr(topology, "name", topology) for topology in mesh.topologies)


def make_mesh(*meshes: Mesh) -> Mesh:
    """Compose nested mesh scopes, preserving slices in device numbering."""
    if not meshes:
        raise ValueError("make_mesh requires at least one mesh")
    result = meshes[0]
    for inner in meshes[1:]:
        here, there = _named(result), _named(inner)
        if set(here).isdisjoint(there):
            result = _joined(
                (*result.topologies, *inner.topologies),
                (*levels(result), *levels(inner)),
                (*starts(result), *starts(inner)),
                (*result.names, *inner.names),
                sliced=isinstance(result.layout, ComposedLayout)
                or isinstance(inner.layout, ComposedLayout),
            )
        elif set(here) <= set(there):
            result = inner
        elif len(there) < len(here) and here[-len(there) :] == there:
            current = result
            kept = len(here) - len(there)
            above = levels(result)[:kept]
            named = sum(len(flatten(level.shape)) for level in above)
            result = _joined(
                (*result.topologies[:kept], *inner.topologies),
                (*above, *levels(inner)),
                (*starts(result)[:kept], *starts(inner)),
                (*result.names[:named], *inner.names),
                sliced=isinstance(result.layout, ComposedLayout)
                or isinstance(inner.layout, ComposedLayout),
            )
            if not within_scope(result, current):
                parent_runs = {
                    name: selected_run(arrangement, start)
                    for name, arrangement, start in zip(
                        here, levels(current), starts(current)
                    )
                    if name in there
                }
                inner_runs = {
                    name: selected_run(arrangement, start)
                    for name, arrangement, start in zip(
                        there, levels(inner), starts(inner)
                    )
                }
                raise ValueError(
                    f"replacement scope selects runs {inner_runs}, outside parent "
                    f"scope runs {parent_runs}; both must be continuous and each "
                    "replacement run must be contained in its parent run"
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


def separate(mesh: Mesh) -> tuple[Mesh, ...]:
    """Split a mesh into one mesh per topology, retaining per-level slices."""
    sliced = isinstance(mesh.layout, ComposedLayout)
    names_at = 0
    separated: list[Mesh] = []
    for topology, level, start in zip(mesh.topologies, levels(mesh), starts(mesh)):
        axis_count = len(flatten(level.shape))
        names = mesh.names[names_at : names_at + axis_count]
        layout: Layout | ComposedLayout = level
        if sliced:
            layout = ComposedLayout(None, start, level)
        separated.append(Mesh((topology,), layout, names))
        names_at += axis_count
    return tuple(separated)


__all__ = [
    "Mesh",
    "Topology",
    "check_topology",
    "levels",
    "make_mesh",
    "selected_run",
    "separate",
    "starts",
    "within_scope",
]
