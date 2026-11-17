"""Meshes: hardware levels, the layout of their units, and sliced meshes."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import services

from .int_tuple import compatible_profile, flatten_to_tuple, unflatten, wrap
from .layout import (
    ComposedLayout,
    Layout,
    LayoutBase,
    domain_offset,
    flatten,
    get,
    plain_layout,
    rank,
    size,
)
from .layout import unflatten as layout_unflatten
from .layout_algebra import make_composed_layout, make_layout
from .stride import compact_row_major
from .topology import Topology, topology_name

__all__ = ["Mesh", "check_topology", "topology_sizes"]


@dataclass(frozen=True)
class Mesh:
    """Describe hardware levels, the layout of their units, and axis names.

    ``layout`` holds one mode per topology and maps a coord to the unit's index
    over all topologies of the mesh, ``T = make_layout(sizes)``. Topology ``t``'s
    mode is ``get(plain_layout(layout), t)``, its strides in that index. A constant
    slice wraps the whole layout in one ``ComposedLayout``; the index of the first
    unit is ``layout(0)``.

    See [shard §5](docs/spec/shard.md#5-mesh).
    """

    topologies: tuple[Topology | str, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Group the layout one mode per topology and refuse an unnamed extent."""
        if len(self.topologies) > 1 and any(
            not isinstance(topology, Topology) for topology in self.topologies
        ):
            raise ValueError("a multi-level Mesh requires Topology values")
        names = tuple(map(topology_name, self.topologies))
        if len(set(names)) != len(names):
            raise ValueError(f"Mesh topology names must be unique, got {names!r}")
        object.__setattr__(self, "layout", nested_layout(self.layout, tuple(self.topologies)))
        for axis, extent in enumerate(flatten_to_tuple(self.layout.shape)):
            if extent is None:
                raise ValueError(
                    f"Mesh: layout axis {axis} must have an explicit extent; "
                    "None is not a ShapeDim. The rule: tilefoundry spec shard mesh"
                )
        check_strides(self)

    @property
    def shape(self) -> tuple:
        """The extent of each mesh axis."""
        return flatten_to_tuple(self.layout.shape)

    def __getitem__(self, key) -> "Mesh":
        """Return a constant sliced mesh selected by integers or step-1 slices.

        Missing axes are full slices; integers select extent one. The box is
        ``make_layout`` of the selected extents with the mesh's strides, moved to
        its first coord by CuTe ``domain_offset``; the index that leaves becomes the
        ``ComposedLayout`` offset. Topologies and names are kept.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        stated = plain_layout(self.layout)
        profile = tuple(flatten_to_tuple(mode) for mode in wrap(stated.shape))
        extents = flatten_to_tuple(stated.shape)
        strides = flatten_to_tuple(stated.strides)
        count = len(extents)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > count:
            raise ValueError(f"mesh slice has {len(keys)} indices but the mesh has {count} axes")
        keys = keys + (slice(None),) * (count - len(keys))
        starts, selected = [], []
        for axis, (k, extent, stride) in enumerate(zip(keys, extents, strides)):
            if not isinstance(extent, int) or not isinstance(stride, int):
                raise ValueError(f"cannot slice mesh axis {axis} with a dynamic extent/stride")
            if isinstance(k, int):
                start = from_end(k, extent)
                if not (0 <= start < extent):
                    raise ValueError(
                        f"mesh slice index {k} out of range for axis {axis} (extent {extent})"
                    )
                stop = start + 1
            elif isinstance(k, slice):
                if k.step not in (None, 1):
                    raise ValueError(f"mesh slice step must be 1 (axis {axis})")
                start = 0 if k.start is None else from_end(k.start, extent)
                stop = extent if k.stop is None else from_end(k.stop, extent)
                if not (0 <= start <= stop <= extent):
                    raise ValueError(
                        f"mesh slice {k.start}:{k.stop} out of range for axis "
                        f"{axis} (extent {extent})"
                    )
                if stop == start:
                    raise ValueError(f"mesh slice selects an empty range on axis {axis}")
            else:
                raise ValueError(f"mesh slice index must be int or slice, got {type(k).__name__}")
            starts.append(start)
            selected.append(stop - start)
        box = make_layout(unflatten(tuple(selected), profile), unflatten(strides, profile))
        box, index = domain_offset(unflatten(tuple(starts), profile), box)
        return Mesh(self.topologies, make_composed_layout(None, index, box), self.names)


def nested_layout(layout_value, topologies: tuple) -> "Layout | ComposedLayout":
    """What a mesh was written as, with every topology's axes under its own mode.

    A mesh naming several topologies may state a tuple of extents or one layout
    over all of their axes. The axes go to the topologies left to right, each
    taking them until their extents multiply to its own size (CuTe ``unflatten``);
    a boundary no prefix of axes lands on is refused. A mesh naming one topology
    holds all axes as its one mode (CuTe ``make_layout`` of the flat layout).
    Strides stay in the index over all topologies.
    """
    if not isinstance(layout_value, LayoutBase):
        layout_value = make_layout(flatten_to_tuple(layout_value))
    if isinstance(layout_value, ComposedLayout):
        if layout_value.outer is None:
            return layout_value
        return make_composed_layout(
            layout_value.inner, layout_value.offset, nested_layout(layout_value.outer, topologies)
        )
    if rank(layout_value) == len(topologies) and all(
        isinstance(mode, tuple) for mode in wrap(layout_value.shape)
    ):
        return layout_value
    flat = Layout(flatten_to_tuple(layout_value.shape), flatten_to_tuple(layout_value.strides))
    if len(topologies) == 1:
        return make_layout(flat)
    profile = compatible_profile(flat.shape, topology_sizes(topologies))
    return layout_unflatten(flat, tuple(wrap(mode) for mode in profile))


def topology_sizes(topologies: tuple) -> tuple:
    """The size of each topology: the shape of ``T``, the index over all of them.

    The outermost topology's size enters no stride of ``T``, so it may be symbolic.
    Every other size MUST be a positive static int.
    """
    for topology in topologies[1:]:
        count = topology.size
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"mesh level {topology.name!r} states extent {count!r}; the index "
                "over all topologies needs the unit count of every inner topology"
            )
    return tuple(getattr(topology, "size", None) for topology in topologies)


def check_strides(mesh: Mesh) -> None:
    """Refuse a topology stride that is not a multiple of that topology's stride in ``T``.

    Topology ``t``'s unit id is ``idx2crd(index, sizes)[t]``; a stride between two
    multiples of ``stride(T)[t]`` would reach units of the topologies inside it. A
    mode of extent 1 never steps, so its stride is not read.
    """
    stated = plain_layout(mesh.layout)
    if len(mesh.topologies) == 1 or stated is None:
        return
    units = compact_row_major(topology_sizes(mesh.topologies))
    for t, (topology, unit) in enumerate(zip(mesh.topologies, units)):
        mode = flatten(get(stated, t))
        for extent, stride in zip(flatten_to_tuple(mode.shape), flatten_to_tuple(mode.strides)):
            if extent == 1 or unit in (0, 1):
                continue
            if isinstance(stride, bool):
                raise ValueError(f"mesh steps by {stride!r}, which is not a dimension")
            if isinstance(stride, int):
                multiple = stride % unit == 0
            else:
                multiple = services.normalize_dim(services.simplify_dim("mod", (stride, unit))) == 0
            if not multiple:
                raise ValueError(
                    f"mesh topology {topology.name!r} steps by {stride!r}, which is not a "
                    f"multiple of its unit {unit} in the index over all topologies"
                )


def check_topology(mesh: Mesh) -> None:
    """Refuse a mesh whose topology modes do not fit their topologies.

    Topology ``t``'s mode MUST NOT hold more units than its static size states,
    ``size(get(P, t)) <= sizes[t]``, and its strides MUST be multiples of
    ``stride(T)[t]``.
    """
    check_strides(mesh)
    stated = plain_layout(mesh.layout)
    if stated is None:
        return
    for t, topology in enumerate(mesh.topologies):
        declared = getattr(topology, "size", None)
        if not isinstance(declared, int) or isinstance(declared, bool):
            continue
        count = size(get(stated, t))
        if isinstance(count, int) and count > declared:
            raise ValueError(
                f"mesh level {topology_name(topology)!r} has {count} "
                f"units, exceeding declared extent {declared}"
            )


def from_end(index: int, extent: int) -> int:
    """A Python index counted from the end, as an index from the start."""
    return index + extent if index < 0 else index
