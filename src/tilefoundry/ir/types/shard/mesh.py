from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard.int_tuple import flatten
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.layout_algebra import c_order_strides


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


def level_axes(mesh: "Mesh") -> tuple[tuple[int, ...], ...]:
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


@lru_cache(maxsize=None)
def positions_at(mesh: Mesh, level: str) -> tuple[tuple, tuple]:
    """Return one named level's shape and normalized strides."""
    names = tuple(topology.name for topology in mesh.topologies)
    if level not in names:
        raise ValueError(f"mesh names levels {names}, not {level!r}")
    shape, strides, _offset = _positions_layout(mesh)
    if any(stride is None for stride in strides):
        strides = c_order_strides(shape)
    index = names.index(level)
    axes = level_axes(mesh)[index]
    below = 1
    for topology in mesh.topologies[index + 1 :]:
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"mesh level {topology.name!r} has a symbolic level below {level!r}"
            )
        below *= topology.size
    non_unit_axes = tuple(axis for axis in axes if shape[axis] != 1)
    normalized = tuple(strides[axis] // below for axis in non_unit_axes)
    return tuple(shape[axis] for axis in non_unit_axes), normalized


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
            if any(
                not isinstance(stride, int) or isinstance(stride, bool)
                for stride in strides
            ):
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
    for topology, axes in zip(mesh.topologies, level_axes(mesh)):
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


def level_projection(mesh: "Mesh", level: str) -> Layout:
    """The layout of the positions *level* has, out of a mesh that names more.

    A position at one level is a position within its parent, so the projection
    keeps every axis up to and including that level's own and divides their
    strides by what the deeper levels contribute. Asking a single-level mesh
    returns its own layout untouched.
    """
    names = tuple(topology.name for topology in mesh.topologies)
    if level not in names:
        raise ValueError(f"mesh names levels {names}, not {level!r}")
    if len(mesh.topologies) == 1:
        if isinstance(mesh.layout, Layout):
            return mesh.layout
        raise ValueError("a sliced mesh states its own layout; it is not projected")
    if not isinstance(mesh.layout, Layout):
        raise ValueError(
            "a mesh naming several levels cannot also be sliced; the slice and the "
            "level boundary would both be deciding which positions these are"
        )
    segments = level_axes(mesh)
    index = names.index(level)
    deeper = 1
    for topology in mesh.topologies[index + 1 :]:
        deeper *= topology.size
    axes = tuple(item for segment in segments[: index + 1] for item in segment)
    shape = flatten(mesh.layout.shape)
    strides = flatten(mesh.layout.strides)
    kept: list[int] = []
    for item in axes:
        stride = strides[item]
        if not isinstance(stride, int) or isinstance(stride, bool) or stride % deeper:
            raise ValueError(
                f"mesh axis {item} has stride {stride!r}, which the {deeper} positions "
                f"below {level!r} do not divide; its positions are not that level's"
            )
        kept.append(stride // deeper)
    return Layout(shape=tuple(shape[item] for item in axes), strides=tuple(kept))


__all__ = [
    "Mesh",
    "Topology",
    "composed",
    "level_axes",
    "level_projection",
    "positions_at",
    "check_topology",
]
