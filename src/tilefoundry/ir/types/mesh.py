from __future__ import annotations

import math
from dataclasses import dataclass

from tilefoundry.ir.types.layout import ComposedLayout, Layout, LayoutBase, flatten, get
from tilefoundry.ir.types.layout import rank as _rank
from tilefoundry.ir.types.stride import compact_row_major, try_compact_major
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
        ``ComposedLayout``. Only a mesh naming one level is sliced: a slice and
        a level boundary would otherwise both decide which positions these are.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if len(self.topologies) != 1:
            raise ValueError("cannot slice a mesh that names several levels")
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        level = get(self.layout, 0)
        shape = level.shape
        strides = level.strides
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
                outer=Layout(shape=(tuple(sub_shape),), strides=(tuple(strides),)),
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
        if len(topologies) != 1:
            raise ValueError(
                "a mesh naming several levels states one arrangement per level; a "
                "slice and a level boundary cannot both decide which positions these are"
            )
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
        size = getattr(topology, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(
                f"mesh level {getattr(topology, 'name', topology)!r} states extent "
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


def make_mesh(

    layout_shape: tuple,
    names: "tuple[str, ...] | None" = None,
    topology: "str | Topology" = "gpu",
) -> Mesh:
    """Convenience constructor for a ``Mesh`` with the given axis extents and C-order strides.

    Convenience constructor for a ``Mesh`` with the given (logical) axis
    extents and C-order strides. ``names`` defaults to ``a, b, c, ...`` (or
    ``g`` for a single axis) so a caller states only the extents instead of
    hand-building a ``Mesh``.

    ``topology`` accepts an explicit ``Topology`` or the ``"gpu"``-shorthand
    default; a raw string is resolved here into a real ``Topology`` sized to
    the domain.
    """
    if names is None:
        names = ("g",) if len(layout_shape) == 1 else tuple("abcdef"[: len(layout_shape)])
    if isinstance(topology, str):
        topology = Topology(topology, math.prod(layout_shape))
    layout_shape = tuple(layout_shape)
    return Mesh(
        topologies=(topology,),
        layout=Layout(shape=layout_shape, strides=try_compact_major(layout_shape)),
        names=tuple(names),
    )


__all__ = ["Mesh", "Topology", "make_mesh"]
