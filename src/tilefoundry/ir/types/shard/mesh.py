from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.layout_algebra import (
    c_order_strides,
    try_c_order_strides,
    unflatten,
)
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

    A constant slice replaces ``layout`` with a ``ComposedLayout`` whose
    ``offset`` and ``outer`` describe the selected sub-box. It remains a
    compile-time descriptor outside the IR/SSA graph.

    See [shard §5](docs/spec/shard.md#5-mesh).
    """

    topologies: tuple[Topology | str, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        layout = self.layout
        if isinstance(layout, tuple):
            layout = Layout(shape=layout, strides=c_order_strides(layout))
            object.__setattr__(self, "layout", layout)

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
        ``ComposedLayout``. Dynamic layouts and nested slices are rejected.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        shape = self.layout.shape
        strides = self.layout.strides
        rank = len(shape)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > rank:
            raise ValueError(f"mesh slice has {len(keys)} indices but the mesh has {rank} axes")
        keys = keys + (slice(None),) * (rank - len(keys))

        sub_shape: list[int] = []
        offset = 0
        for axis, (k, extent, stride) in enumerate(zip(keys, shape, strides)):
            if not isinstance(extent, int) or not isinstance(stride, int):
                raise ValueError(f"cannot slice mesh axis {axis} with a dynamic extent/stride")
            if isinstance(k, int):
                start = k + extent if k < 0 else k
                if not (0 <= start < extent):
                    raise ValueError(
                        f"mesh slice index {k} out of range for axis {axis} (extent {extent})"
                    )
                sel = 1
            elif isinstance(k, slice):
                if k.step not in (None, 1):
                    raise ValueError(f"mesh slice step must be 1 (axis {axis})")
                start = 0 if k.start is None else (k.start + extent if k.start < 0 else k.start)
                stop = extent if k.stop is None else (k.stop + extent if k.stop < 0 else k.stop)
                if not (0 <= start <= stop <= extent):
                    raise ValueError(
                        f"mesh slice {k.start}:{k.stop} out of range for axis {axis} (extent {extent})"
                    )
                sel = stop - start
                if sel == 0:
                    raise ValueError(f"mesh slice selects an empty range on axis {axis}")
            else:
                raise ValueError(f"mesh slice index must be int or slice, got {type(k).__name__}")
            offset += start * stride
            sub_shape.append(sel)

        sliced = ComposedLayout(
            inner=None,
            offset=offset,
            outer=Layout(shape=tuple(sub_shape), strides=strides),
        )
        return Mesh(
            topologies=self.topologies,
            layout=sliced,
            names=self.names,
        )


def topology_axes(mesh: "Mesh") -> tuple[tuple[int, ...], ...]:
    """Which of *mesh*'s layout axes belong to each level it names, in order.

    The axes are handed to the levels left to right, and a level takes them
    until their extents multiply to exactly its own size. A boundary that no
    prefix of axes lands on is refused rather than guessed: one axis of four
    positions across a two-CTA and two-thread boundary could be either half of
    it, and picking one would place work somewhere nobody said.

    A mesh naming one level owns all of them, which is the answer this gives
    without any of the arithmetic.
    """
    if len(mesh.topologies) == 1:
        return (tuple(range(len(flatten(mesh.layout.shape)))),)
    extents = flatten(mesh.layout.shape)
    found: list[tuple[int, ...]] = []
    axis = 0
    for topology in mesh.topologies:
        size = topology.size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(
                f"mesh level {topology.name!r} states extent {size!r}; segmenting a "
                "mesh that names several levels needs each of their position counts"
            )
        taken: list[int] = []
        product = 1
        while product < size and axis < len(extents):
            extent = extents[axis]
            if not isinstance(extent, int) or isinstance(extent, bool):
                raise ValueError(
                    f"mesh layout axis {axis} states extent {extent!r}; segmenting a "
                    "mesh that names several levels needs concrete axis extents"
                )
            product *= extent
            taken.append(axis)
            axis += 1
        if product != size:
            raise ValueError(
                f"mesh axes {tuple(extents)} do not land on the boundary of level "
                f"{topology.name!r} at {size}: the axes up to there multiply to "
                f"{product}. Write the axis that straddles it as the two axes it is"
            )
        found.append(tuple(taken))
    if axis != len(extents):
        raise ValueError(
            f"mesh layout has {len(extents)} axes but the levels it names account "
            f"for {axis}; every axis belongs to one of them"
        )
    return tuple(found)


def _positions_layout(mesh: Mesh) -> tuple[tuple, tuple, int]:
    """Return flattened shape, strides, and offset for a supported mesh layout."""
    if isinstance(mesh.layout, Layout):
        return flatten(mesh.layout.shape), flatten(mesh.layout.strides), 0
    if mesh.layout.inner is None and isinstance(mesh.layout.outer, Layout):
        return (
            flatten(mesh.layout.outer.shape),
            flatten(mesh.layout.outer.strides),
            mesh.layout.offset,
        )
    raise ValueError(f"mesh levels {mesh.topologies!r} have an unsupported layout")


def mesh_image(mesh: Mesh) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    """A mesh's positions as one static ``(offset, shape, strides)``, flattened.

    The reading anything counting positions starts from: where the mesh's first
    position sits and what walking each of its axes steps by, as numbers. A
    layout stating no steps is the C order its shape gives it, which is what
    every layer of this IR reads it as.

    ``None`` where the mesh states something no number answers: a composition
    through a transform, or an extent, step or offset that is symbolic.
    """
    from tilefoundry.ir.types.shape_helpers import (  # noqa: PLC0415 - cycle guard
        static_dim_value,
    )

    layout = mesh.layout
    if isinstance(layout, Layout):
        stated, offset = layout, 0
    elif isinstance(layout, ComposedLayout) and layout.inner is None and isinstance(
        layout.outer, Layout
    ):
        stated, offset = layout.outer, layout.offset
    else:
        return None

    shape = tuple(static_dim_value(value) for value in flatten(stated.shape))
    if any(value is None for value in shape):
        return None
    if stated.strides is None:
        strides = try_c_order_strides(shape)
    else:
        strides = tuple(static_dim_value(value) for value in flatten(stated.strides))
        if any(value is None for value in strides):
            strides = None
    static_offset = static_dim_value(offset)
    if strides is None or static_offset is None or len(shape) != len(strides):
        return None
    return static_offset, shape, strides


def _coalesced(modes: list[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One level's modes as a set of positions reads them: sorted by step, joined."""
    joined: list[list[int]] = []
    for extent, step in sorted(modes, key=lambda mode: (mode[1], mode[0])):
        if joined and joined[-1][0] * joined[-1][1] == step:
            joined[-1][0] *= extent
        else:
            joined.append([extent, step])
    return tuple(extent for extent, _ in joined), tuple(step for _, step in joined)


def level_positions(
    mesh: Mesh,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] | None:
    """Each level's modes and offset, in that level's own numbering.

    A mode belongs to the level its step falls in, split where it crosses a
    boundary, so a level takes the positions its steps say are its own rather
    than the axes standing where it does. A sub-mesh -- one warp of a CTA's
    threads -- is a level whose modes do not multiply to its size, which is all
    that separates this from :func:`topology_axes`. Modes of one position are
    left out and the rest sorted by step and joined. ``None`` where a step, a
    mode crossing a boundary, a level's end or a size is not read.
    """
    image = mesh_image(mesh)
    if image is None:
        return None
    offset, extents, strides = image
    names = tuple(getattr(topology, "name", topology) for topology in mesh.topologies)
    sizes = tuple(getattr(topology, "size", None) for topology in mesh.topologies)
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in sizes
    ):
        if len(sizes) != 1:
            return None
        held = [
            (extent, stride)
            for extent, stride in zip(extents, strides)
            if extent != 1 and stride != 0
        ]
        return {names[0]: (*_coalesced(held), offset)}
    below: list[int] = []
    count = 1
    for size in reversed(sizes):
        below.insert(0, count)
        count *= size
    modes: list[list[tuple[int, int]]] = [[] for _ in sizes]
    pending = [
        (extent, stride)
        for extent, stride in zip(extents, strides)
        if extent != 1 and stride != 0
    ]
    while pending:
        extent, stride = pending.pop()
        level = next(
            (
                index
                for index, (unit, size) in enumerate(zip(below, sizes))
                if unit <= stride < unit * size
            ),
            None,
        )
        if level is None or stride % below[level]:
            return None
        step, size = stride // below[level], sizes[level]
        if extent * step <= size:
            modes[level].append((extent, step))
            continue
        inside = size // step
        if size % step or extent % inside:
            return None
        modes[level].append((inside, step))
        pending.append((extent // inside, below[level] * size))
    found: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {}
    for name, unit, size, held in zip(names, below, sizes, modes):
        start = (offset // unit) % size
        if start + sum((extent - 1) * step for extent, step in held) >= size:
            return None
        found[name] = (*_coalesced(held), start)
    return found


@lru_cache(maxsize=None)
def grouped_layout(mesh: Mesh) -> Layout:
    """*mesh*'s positions with one mode per level it names, in mesh numbering.

    The grouping only says which axes are whose; the strides still count
    positions of the whole mesh. What one level alone would call them is that
    mode divided by :func:`positions_below`.
    """
    shape, strides, _offset = _positions_layout(mesh)
    if any(stride is None for stride in strides):
        strides = c_order_strides(shape)
    profile = topology_axes(mesh)
    return Layout(shape=unflatten(shape, profile), strides=unflatten(strides, profile))


def positions_below(mesh: Mesh, index: int) -> int:
    """How many positions the levels under the one at *index* contribute."""
    below = 1
    for topology in mesh.topologies[index + 1 :]:
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"mesh level {mesh.topologies[index].name!r} has a symbolic level below it"
            )
        below *= topology.size
    return below


def _index_of(mesh: Mesh, topology_level: str) -> int:
    names = tuple(topology.name for topology in mesh.topologies)
    if topology_level not in names:
        raise ValueError(f"mesh names levels {names}, not {topology_level!r}")
    return names.index(topology_level)


def _divided(strides: tuple, below: int, topology_level: str) -> tuple[int, ...]:
    """*strides* read as the level's own, or a refusal that they are not."""
    divided: list[int] = []
    for axis, stride in enumerate(strides):
        if not isinstance(stride, int) or isinstance(stride, bool) or stride % below:
            raise ValueError(
                f"mesh axis {axis} has stride {stride!r}, which the {below} positions "
                f"below {topology_level!r} do not divide; its positions are not that level's"
            )
        divided.append(stride // below)
    return tuple(divided)


def positions_at(mesh: Mesh, topology_level: str) -> tuple[tuple, tuple]:
    """Return one named level's shape and normalized strides, every axis of it.

    An axis of one position is still that level's axis: dropping it here would
    leave the level's own layout narrower than the attrs written against those
    axes, and nothing downstream could say which attr went with which mode.
    """
    index = _index_of(mesh, topology_level)
    grouped = grouped_layout(mesh)
    below = positions_below(mesh, index)
    shape = flatten(grouped.shape[index])
    return shape, _divided(flatten(grouped.strides[index]), below, topology_level)


def _suffix_refine(outer: Mesh, inner: Mesh) -> "Mesh | None":
    """*outer* with the levels *inner* names replaced, the ones above it kept.

    A scope naming a suffix of the levels in force narrows those levels and
    says nothing about the ones above, so those stay exactly as they were: a
    grid of CTAs each running the same program over some of its threads is one
    mesh, not a choice between naming the grid and naming the threads.

    ``None`` when *inner* does not name a strict suffix of *outer*'s levels,
    which leaves the composition to the rules that do apply. A level of one
    position takes an axis of *outer* only where the mesh wrote one down.
    """
    outer_names = tuple(getattr(level, "name", level) for level in outer.topologies)
    inner_names = tuple(getattr(level, "name", level) for level in inner.topologies)
    if not inner_names or len(inner_names) >= len(outer_names):
        return None
    if outer_names[-len(inner_names) :] != inner_names:
        return None

    prefix = len(outer_names) - len(inner_names)
    outer_image, inner_image = mesh_image(outer), mesh_image(inner)
    if outer_image is None or inner_image is None:
        raise ValueError(
            f"composing {inner_names} into {outer_names} needs both scopes' "
            "positions as numbers"
        )
    outer_offset, outer_shape, outer_strides = outer_image
    inner_offset, inner_shape, inner_strides = inner_image

    def positions(topologies) -> int:
        count = 1
        for topology in topologies:
            size = getattr(topology, "size", None)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(
                    f"mesh level {getattr(topology, 'name', topology)!r} states extent "
                    f"{size!r}; composing scopes needs each level's position count"
                )
            count *= size
        return count

    above = positions(outer.topologies[:prefix])
    inside = positions(outer.topologies[prefix:])
    taken, product = 0, 1
    while taken < len(outer_shape) and product < above:
        product *= outer_shape[taken]
        taken += 1
    if above == 1 and outer_shape and outer_shape[0] == 1:
        taken = 1
    if product != above:
        raise ValueError(
            f"composing {inner_names} into {outer_names} cannot tell which axes "
            f"of {outer_shape} are {outer_names[:prefix]}'s {above} positions"
        )

    return Mesh(
        topologies=(*outer.topologies[:prefix], *inner.topologies),
        layout=ComposedLayout(
            inner=None,
            offset=(outer_offset // inside) * inside + inner_offset,
            outer=Layout(
                shape=(*outer_shape[:taken], *inner_shape),
                strides=(*outer_strides[:taken], *inner_strides),
            ),
        ),
        names=(*outer.names[:taken], *inner.names),
    )


def composed(meshes: "tuple[Mesh, ...]") -> "Mesh":
    """Compose scopes, replacing an existing level when the inner names it."""
    if len(meshes) == 1:
        check_topology(meshes[0])
        return meshes[0]

    def positions(mesh: Mesh) -> int:
        count = 1
        for topology in mesh.topologies:
            size = topology.size
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(
                    f"mesh level {topology.name!r} states extent {size!r}; "
                    "composing scopes needs each level's position count"
                )
            count *= size
        return count

    def concatenate(outer: Mesh, inner: Mesh) -> Mesh:
        outer_shape, outer_strides, outer_offset = _positions_layout(outer)
        inner_shape, inner_strides, inner_offset = _positions_layout(inner)
        for mesh, strides in ((outer, outer_strides), (inner, inner_strides)):
            if any(not isinstance(stride, int) or isinstance(stride, bool) for stride in strides):
                raise ValueError(
                    f"mesh levels {mesh.topologies!r} need concrete strides to compose"
                )
        below = positions(inner)
        layout = Layout(
            shape=(*outer_shape, *inner_shape),
            strides=(*(stride * below for stride in outer_strides), *inner_strides),
        )
        offset = outer_offset * below + inner_offset
        sliced = isinstance(outer.layout, ComposedLayout) or isinstance(
            inner.layout, ComposedLayout
        )
        return Mesh(
            topologies=(*outer.topologies, *inner.topologies),
            layout=(
                layout if not sliced else ComposedLayout(inner=None, offset=offset, outer=layout)
            ),
            names=(*outer.names, *inner.names),
        )

    result = meshes[0]
    for inner in meshes[1:]:
        refined = _suffix_refine(result, inner)
        if refined is not None:
            result = refined
            continue
        current_names = {topology.name for topology in result.topologies}
        inner_names = {topology.name for topology in inner.topologies}
        if current_names.isdisjoint(inner_names):
            result = concatenate(result, inner)
            continue
        if current_names <= inner_names:
            result = inner
            continue
        raise ValueError(
            f"{sorted(current_names & inner_names)} named again while "
            f"{sorted(current_names - inner_names)} is not; a scope either "
            "replaces the levels in force or adds levels below them"
        )
    check_topology(result)
    return result


def check_topology(mesh: Mesh) -> None:
    """Reject static mesh positions beyond their declared topology extents.

    A constant slice is already bounded by ``Mesh.__getitem__``; its shortened
    axes no longer land on full topology boundaries and are therefore accepted.
    """
    if isinstance(mesh.layout, ComposedLayout):
        return
    shape, _strides, _offset = _positions_layout(mesh)
    for topology, axes in zip(mesh.topologies, topology_axes(mesh)):
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            continue
        count = 1
        for axis in axes:
            extent = shape[axis]
            if not isinstance(extent, int) or isinstance(extent, bool):
                count = None
                break
            count *= extent
        if count is not None and count > topology.size:
            raise ValueError(
                f"mesh level {topology.name!r} has {count} positions, exceeding declared extent {topology.size}"
            )


def topology_projection(mesh: "Mesh", topology_level: str) -> Layout:
    """The layout of the positions *level* has, out of a mesh that names more.

    A position at one level is a position within its parent, so the projection
    keeps every axis up to and including that level's own and divides their
    strides by what the deeper levels contribute. Asking a single-level mesh
    returns its own layout untouched.
    """
    index = _index_of(mesh, topology_level)
    if len(mesh.topologies) == 1:
        if isinstance(mesh.layout, Layout):
            return mesh.layout
        raise ValueError("a sliced mesh states its own layout; it is not projected")
    if not isinstance(mesh.layout, Layout):
        raise ValueError(
            "a mesh naming several levels cannot also be sliced; the slice and the "
            "level boundary would both be deciding which positions these are"
        )
    grouped = grouped_layout(mesh)
    below = positions_below(mesh, index)
    shape = flatten(grouped.shape[: index + 1])
    strides = _divided(flatten(grouped.strides[: index + 1]), below, topology_level)
    return Layout(shape=shape, strides=strides)


__all__ = [
    "Mesh",
    "Topology",
    "composed",
    "grouped_layout",
    "level_positions",
    "mesh_image",
    "positions_at",
    "positions_below",
    "topology_axes",
    "topology_projection",
    "check_topology",
]
