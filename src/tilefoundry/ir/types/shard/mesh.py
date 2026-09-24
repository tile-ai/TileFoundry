from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.shard.int_tuple import flatten, product
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase
from tilefoundry.ir.types.shard.layout_algebra import c_order_strides
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
    layout: "LayoutBase | tuple[LayoutBase, ...]"
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _as_written(self.layout, len(self.topologies)))
        if len(self.topologies) > 1 and not isinstance(self.layout, tuple):
            object.__setattr__(
                self, "layout", _as_levels(self.layout, tuple(self.topologies))
            )
        for axis, extent in enumerate(flatten(self.written.shape)):
            if extent is None:
                raise ValueError(
                    f"Mesh: layout axis {axis} must have an explicit extent; "
                    "None is not a ShapeDim. The rule: tilefoundry spec shard mesh"
                )

    @property
    def levels(self) -> tuple[LayoutBase, ...]:
        """One arrangement per level it names, each in that level's own numbering.

        A mesh naming several levels states one arrangement per level from the
        moment it is built; one naming a single level states that level's
        arrangement directly, and is read as the one-level tuple it is.
        """
        return self.layout if isinstance(self.layout, tuple) else (self.layout,)

    @property
    def positions(self) -> Layout:
        """Every level's axes end to end, as the device numbers the positions.

        Each level states its own numbering, and a position of one level is a
        position within its parent, so an axis steps by what it states times
        what the levels under it hold. Where a level states a run rather than
        all of its positions, where that run starts is :attr:`offset`.
        """
        if isinstance(self.layout, LayoutBase):
            stated = stated_layout(self.layout)
            return Layout(
                shape=tuple(flatten(stated.shape)), strides=tuple(flatten(stated.strides))
            )
        shape: list = []
        strides: list = []
        for index, arrangement in enumerate(self.layout):
            stated = stated_layout(arrangement)
            below = positions_below(self, index)
            shape.extend(flatten(stated.shape))
            strides.extend(stride * below for stride in flatten(stated.strides))
        return Layout(shape=tuple(shape), strides=tuple(strides))

    @property
    def written(self) -> LayoutBase:
        """The mesh's positions as one statement, the way it was written down.

        The whole mesh's :attr:`positions`, with a run's start in the offset of
        a composition where any level states one -- which is where the

        Each level states its own numbering, and a position of one level is a
        position within its parent, so an axis steps by what it states times
        what the levels under it hold. That product is the one numbering a
        statement it was written as had it.
        """
        if isinstance(self.layout, LayoutBase):
            return self.layout
        whole = self.positions
        if not self.sliced:
            return whole
        return ComposedLayout(inner=None, offset=self.offset, outer=whole)


    @property
    def sliced(self) -> bool:
        """Whether any level states a run rather than all of its positions."""
        stated = (self.layout,) if isinstance(self.layout, LayoutBase) else self.layout
        return any(isinstance(arrangement, ComposedLayout) for arrangement in stated)

    @property
    def offset(self) -> int:
        """Where the whole mesh's first position sits, as the device numbers it."""
        if isinstance(self.layout, LayoutBase):
            return self.layout.offset if isinstance(self.layout, ComposedLayout) else 0
        total = 0
        for index, arrangement in enumerate(self.layout):
            start = arrangement.offset if isinstance(arrangement, ComposedLayout) else 0
            total += start * positions_below(self, index)
        return total

    def __getitem__(self, key) -> "Mesh":
        """Return a constant sub-mesh selected by integers or unit-step slices.

        Missing axes are full slices; integers select extent one. The result
        preserves topology and names while recording the sub-box as a
        ``ComposedLayout``. Only a mesh naming one level is sliced: a slice and
        a level boundary would otherwise both decide which positions these are.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if len(self.topologies) != 1:
            raise ValueError("cannot slice a mesh that names several levels")
        stated = self.layout if isinstance(self.layout, LayoutBase) else self.layout[0]
        if isinstance(stated, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        shape = stated.shape
        strides = stated.strides
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
                        f"mesh slice {k.start}:{k.stop} out of range for axis "
                        f"{axis} (extent {extent})"
                    )
                sel = stop - start
                if sel == 0:
                    raise ValueError(f"mesh slice selects an empty range on axis {axis}")
            else:
                raise ValueError(f"mesh slice index must be int or slice, got {type(k).__name__}")
            offset += start * stride
            sub_shape.append(sel)

        return Mesh(
            topologies=self.topologies,
            layout=ComposedLayout(
                inner=None,
                offset=offset,
                outer=Layout(shape=tuple(sub_shape), strides=strides),
            ),
            names=self.names,
        )


def stated_layout(arrangement: LayoutBase) -> Layout:
    """One level's arrangement as the extents and steps it states.

    A level reached through a transform states no steps of its own and is
    refused rather than read as the affine half of something it is not; one
    stating no steps is the C order its extents give it.
    """
    if isinstance(arrangement, ComposedLayout):
        if arrangement.inner is not None or not isinstance(arrangement.outer, Layout):
            raise ValueError(f"mesh level {arrangement!r} states no extents and steps")
        arrangement = arrangement.outer
    if not isinstance(arrangement, Layout):
        raise ValueError(f"mesh level {arrangement!r} is not an arrangement")
    if arrangement.strides is None:
        return Layout(shape=arrangement.shape, strides=c_order_strides(arrangement.shape))
    return arrangement


def _as_written(layout, levels: int) -> "LayoutBase | tuple[LayoutBase, ...]":
    """What the author stated, as arrangements: one, or one per level."""
    if isinstance(layout, LayoutBase):
        return layout
    if isinstance(layout, tuple) and layout and all(
        isinstance(item, LayoutBase) for item in layout
    ):
        return tuple(layout)
    if levels > 1 and isinstance(layout, tuple) and len(layout) == levels and all(
        isinstance(item, tuple) for item in layout
    ):
        return tuple(
            Layout(shape=tuple(item), strides=c_order_strides(tuple(item))) for item in layout
        )
    extents = tuple(layout)
    return Layout(shape=extents, strides=c_order_strides(extents))


def _as_levels(layout, topologies: tuple) -> tuple[LayoutBase, ...]:
    """Read what a caller wrote as one arrangement per level.

    A mesh naming one level states that level's arrangement directly. One
    naming several may state a tuple of them, or one arrangement over all of
    their axes: the axes are then handed to the levels left to right, each
    level taking them until their extents multiply to its own size, and each
    level's steps are divided by what the levels under it hold so that what
    comes back is the arrangement that level would have alone. A boundary no
    prefix of axes lands on is refused rather than guessed.
    """
    if isinstance(layout, tuple):
        return tuple(layout)
    if len(topologies) == 1:
        return (layout,)
    return _segmented(layout, topologies)


def _segmented(stated: LayoutBase, topologies: tuple) -> tuple[LayoutBase, ...]:
    """One arrangement over every level's axes, cut at the level boundaries."""
    if isinstance(stated, ComposedLayout):
        raise ValueError(
            "a mesh naming several levels states one arrangement per level; a "
            "slice and a level boundary cannot both decide which positions these are"
        )
    extents = tuple(flatten(stated.shape))
    steps = tuple(flatten(stated.strides if stated.strides is not None else c_order_strides(extents)))
    below = 1
    sizes = []
    for topology in reversed(topologies):
        sizes.insert(0, below)
        size = getattr(topology, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(
                f"mesh level {getattr(topology, 'name', topology)!r} states extent "
                f"{size!r}; cutting one arrangement at the level boundaries needs "
                "each of their position counts"
            )
        below *= size
    found: list[LayoutBase] = []
    axis = 0
    for topology, unit in zip(topologies, sizes):
        size = topology.size
        taken_extents: list = []
        taken_steps: list = []
        product_so_far = 1
        while product_so_far < size and axis < len(extents):
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
            product_so_far *= extent
            taken_extents.append(extent)
            taken_steps.append(step // unit)
            axis += 1
        if product_so_far != size:
            raise ValueError(
                f"mesh axes {extents} do not land on the boundary of level "
                f"{topology.name!r} at {size}: the axes up to there multiply to "
                f"{product_so_far}. Write the axis that straddles it as the two axes it is"
            )
        found.append(Layout(shape=tuple(taken_extents), strides=tuple(taken_steps)))
    if axis != len(extents):
        raise ValueError(
            f"mesh layout has {len(extents)} axes but the levels it names account "
            f"for {axis}; every axis belongs to one of them"
        )
    return tuple(found)


def level_axes(mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    """Which of *mesh*'s axes belong to each level it names, in ``names`` order.

    The levels state their own arrangements, so this only counts what each one
    wrote down: nothing here decides where a boundary falls.
    """
    found: list[tuple[int, ...]] = []
    axis = 0
    for arrangement in mesh.levels:
        width = len(flatten(arrangement.shape))
        found.append(tuple(range(axis, axis + width)))
        axis += width
    return tuple(found)


def level_index(mesh: Mesh, topology_level: str) -> int:
    """Which of *mesh*'s levels is the one named."""
    names = tuple(getattr(topology, "name", topology) for topology in mesh.topologies)
    if topology_level not in names:
        raise ValueError(f"mesh names levels {names}, not {topology_level!r}")
    return names.index(topology_level)


def positions_below(mesh: Mesh, index: int) -> int:
    """CuTe ``size(take<index + 1, rank>)``: what the levels under this one hold."""
    below = 1
    for topology in mesh.topologies[index + 1 :]:
        size = getattr(topology, "size", None)
        if not isinstance(size, int) or isinstance(size, bool):
            raise ValueError(
                f"mesh level {getattr(mesh.topologies[index], 'name', '?')!r} has a "
                "symbolic level below it"
            )
        below *= size
    return below


def append(outer: Mesh, inner: Mesh) -> Mesh:
    """CuTe ``append``: *inner*'s levels stated below *outer*'s, each its own."""
    return Mesh(
        topologies=(*outer.topologies, *inner.topologies),
        layout=(*outer.levels, *inner.levels),
        names=(*outer.names, *inner.names),
    )


def replace(outer: Mesh, inner: Mesh) -> Mesh:
    """CuTe ``replace``: *inner*'s levels in place of *outer*'s last ones.

    The levels above keep their arrangements and their names untouched, which
    is what a scope naming a suffix of the levels in force states: it narrows
    those levels and says nothing about the ones it did not name.
    """
    kept = len(outer.topologies) - len(inner.topologies)
    named = sum(len(flatten(stated_layout(level).shape)) for level in outer.levels[:kept])
    return Mesh(
        topologies=(*outer.topologies[:kept], *inner.topologies),
        layout=(*outer.levels[:kept], *inner.levels),
        names=(*outer.names[:named], *inner.names),
    )


def merge_mesh(meshes: "tuple[Mesh, ...]") -> "Mesh":
    """The scope in force once each of *meshes* has been entered in turn.

    A scope naming levels none of those in force name is appended below them.
    One naming every level in force replaces them. One naming a suffix of them
    replaces that suffix and keeps what is above. Any other overlap is refused
    rather than decomposed: which positions the half-named levels would then
    state is nobody's statement.
    """
    result = meshes[0]
    for inner in meshes[1:]:
        outer_names = tuple(getattr(level, "name", level) for level in result.topologies)
        inner_names = tuple(getattr(level, "name", level) for level in inner.topologies)
        if set(outer_names).isdisjoint(inner_names):
            result = append(result, inner)
        elif set(outer_names) <= set(inner_names):
            result = inner
        elif (
            len(inner_names) < len(outer_names)
            and outer_names[-len(inner_names) :] == inner_names
        ):
            result = replace(result, inner)
        else:
            shared = sorted(set(outer_names) & set(inner_names))
            unnamed = sorted(set(outer_names) - set(inner_names))
            raise ValueError(
                f"{shared} named again while {unnamed} is not; a scope either "
                "replaces the levels in force or adds levels below them"
            )
    check_topology(result)
    return result


def check_topology(mesh: Mesh) -> None:
    """Reject static mesh positions beyond their declared topology extents.

    A constant slice is already bounded by ``Mesh.__getitem__``; its shortened
    axes no longer land on full topology boundaries and are therefore accepted.
    """
    for topology, arrangement in zip(mesh.topologies, mesh.levels):
        size = getattr(topology, "size", None)
        if isinstance(arrangement, ComposedLayout):
            continue
        if not isinstance(size, int) or isinstance(size, bool):
            continue
        count = product(tuple(flatten(arrangement.shape)))
        if isinstance(count, int) and count > size:
            raise ValueError(
                f"mesh level {getattr(topology, 'name', topology)!r} has {count} "
                f"positions, exceeding declared extent {size}"
            )


__all__ = [
    "Mesh",
    "Topology",
    "append",
    "check_topology",
    "level_axes",
    "level_index",
    "merge_mesh",
    "positions_below",
    "replace",
    "stated_layout",
]
