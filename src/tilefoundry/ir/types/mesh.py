"""Meshes: hardware levels, the layout of their units, and sub-mesh slices."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tilefoundry import services
from tilefoundry.ir import expr

from . import layout
from .int_tuple import compatible_profile
from .layout import (
    ComposedLayout,
    Layout,
    LayoutBase,
    flatten,
    plain_layout,
    size,
    unflatten,
    with_offset,
)
from .layout_algebra import filter, is_contiguous
from .stride import compact_row_major, crd2idx, idx2crd
from .topology import Topology, topology_name

__all__ = [
    "Mesh",
    "axis_keys",
    "check_topology",
    "equals_mesh_scope",
    "flat_layouts",
    "get",
    "layouts",
    "offset",
    "selected_units",
    "sub_box",
    "sub_meshes",
    "topology_names",
    "within_mesh_scope",
]


@dataclass(frozen=True)
class Mesh:
    """Describe hardware levels, the layout of their units, and axis names.

    ``layout`` runs parallel to ``topologies``: one layout per level, in
    that level's own numbering, so a level states which of its units it
    selects without any level above it entering the answer. A constant slice
    wraps the whole layout in one ``ComposedLayout``. Its ``offset`` is the
    index of the first unit over all topologies of the mesh.

    See [shard §5](docs/spec/shard.md#5-mesh).
    """

    topologies: tuple[Topology | str, ...]
    layout: "Layout | ComposedLayout"
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the layout to one layout per level and refuse an unnamed extent."""
        if len(self.topologies) > 1 and any(
            not isinstance(topology, Topology) for topology in self.topologies
        ):
            raise ValueError("a multi-level Mesh requires Topology values")
        names = tuple(map(topology_name, self.topologies))
        if len(set(names)) != len(names):
            raise ValueError(f"Mesh topology names must be unique, got {names!r}")
        object.__setattr__(self, "layout", nested_layout(self.layout, tuple(self.topologies)))
        for axis, extent in enumerate(flatten(self.layout.shape)):
            if extent is None:
                raise ValueError(
                    f"Mesh: layout axis {axis} must have an explicit extent; "
                    "None is not a ShapeDim. The rule: tilefoundry spec shard mesh"
                )

    @property
    def shape(self) -> tuple:
        """The extent of each mesh axis."""
        return tuple(flatten(self.layout).shape)

    def __getitem__(self, key) -> "Mesh":
        """Return a constant sub-mesh selected by integers or step-1 slices.

        Missing axes are full slices; integers select extent one. The result
        preserves topology and names while recording the sub-box as a
        ``ComposedLayout``. The strides and the offset are both in the index
        over all topologies of the mesh.

        See [shard §5](docs/spec/shard.md#5-mesh).
        """
        if isinstance(self.layout, ComposedLayout):
            raise ValueError("cannot slice an already-sliced mesh (nested slice unsupported)")
        stated = plain_layout(self.layout)
        groups = tuple(layout.get(stated, index) for index in range(len(stated.shape)))
        rank = sum(len(flatten(group.shape)) for group in groups)
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > rank:
            raise ValueError(f"mesh slice has {len(keys)} indices but the mesh has {rank} axes")
        keys = keys + (slice(None),) * (rank - len(keys))

        sub_levels: list[Layout] = []
        total = 0
        axis = 0
        for group in groups:
            level_shape = tuple(flatten(group.shape))
            level_strides = tuple(flatten(group.strides))
            sub_shape: list[int] = []
            for k, extent, stride in zip(
                keys[axis : axis + len(level_shape)], level_shape, level_strides
            ):
                if not isinstance(extent, int) or not isinstance(stride, int):
                    raise ValueError(f"cannot slice mesh axis {axis} with a dynamic extent/stride")
                if isinstance(k, int):
                    start = from_end(k, extent)
                    if not (0 <= start < extent):
                        raise ValueError(
                            f"mesh slice index {k} out of range for axis {axis} (extent {extent})"
                        )
                    selected = 1
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
                    selected = stop - start
                    if selected == 0:
                        raise ValueError(f"mesh slice selects an empty range on axis {axis}")
                else:
                    raise ValueError(
                        f"mesh slice index must be int or slice, got {type(k).__name__}"
                    )
                total += start * stride
                sub_shape.append(selected)
                axis += 1
            sub_levels.append(Layout(tuple(sub_shape), level_strides))

        return Mesh(
            topologies=self.topologies,
            layout=ComposedLayout(
                inner=None,
                offset=total,
                outer=joined_layout(tuple(sub_levels)),
            ),
            names=self.names,
        )

    @classmethod
    def from_meshes(cls, *meshes: Mesh) -> Mesh:
        """Compose nested mesh scopes, preserving slices in device numbering."""
        if not meshes:
            raise ValueError("make_mesh requires at least one mesh")
        result = meshes[0]
        for inner in meshes[1:]:
            here, there = topology_names(result), topology_names(inner)
            sliced = isinstance(result.layout, ComposedLayout) or isinstance(
                inner.layout, ComposedLayout
            )
            if set(here).isdisjoint(there):
                result = joined(
                    (*result.topologies, *inner.topologies),
                    (*layouts(result), *layouts(inner)),
                    (*map(offset, sub_meshes(result)), *map(offset, sub_meshes(inner))),
                    (*result.names, *inner.names),
                    sliced=sliced,
                )
            elif set(here) <= set(there):
                result = inner
            elif len(there) < len(here) and here[-len(there) :] == there:
                current = result
                kept = len(here) - len(there)
                above = layouts(result)[:kept]
                named = sum(len(flatten(level.shape)) for level in above)
                result = joined(
                    (*result.topologies[:kept], *inner.topologies),
                    (*above, *layouts(inner)),
                    (
                        *tuple(map(offset, sub_meshes(result)))[:kept],
                        *map(offset, sub_meshes(inner)),
                    ),
                    (*result.names[:named], *inner.names),
                    sliced=sliced,
                )
                if not within_mesh_scope(result, current):
                    raise ValueError(
                        f"replacement scope selects runs {runs(inner, there)}, outside parent "
                        f"scope runs {runs(current, there)}; both must be continuous and each "
                        "replacement selection must be contained in its parent selection"
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


def nested_layout(layout_value, topologies: tuple) -> "Layout | ComposedLayout":
    """What a mesh was written as, with every topology's axes under its own mode.

    A mesh naming several topologies may state a tuple of extents or one layout
    over all of their axes. The axes go to the topologies left to right, each
    taking them until their extents multiply to its own size; a boundary no
    prefix of axes lands on is refused. Strides stay in the index over all
    topologies: mode ``i`` steps by topology ``i``'s own unit id times the
    units of the topologies inside it.
    """
    if not isinstance(layout_value, LayoutBase):
        extents = tuple(flatten(layout_value))
        layout_value = Layout(extents, compact_row_major(extents))
    if isinstance(layout_value, ComposedLayout):
        if layout_value.outer is None:
            return layout_value
        return ComposedLayout(
            inner=layout_value.inner,
            offset=layout_value.offset,
            outer=nested_layout(layout_value.outer, topologies),
        )
    if len(layout_value.shape) == len(topologies) and all(
        isinstance(mode, tuple) for mode in layout_value.shape
    ):
        grouped = layout_value
    else:
        extents = tuple(flatten(layout_value.shape))
        steps = tuple(flatten(layout_value.strides))
        if len(topologies) == 1:
            return Layout(shape=(extents,), strides=(steps,))
        unit_scales(topologies)
        profile = compatible_profile(extents, tuple(t.size for t in topologies))
        grouped = unflatten(
            Layout(extents, steps),
            tuple(mode if isinstance(mode, tuple) else (mode,) for mode in profile),
        )
    for topology, scale, steps in zip(topologies, unit_scales(topologies), grouped.strides):
        scale_strides(steps, scale, topology, divide=True)
    return grouped


def unit_scales(topologies: tuple) -> tuple[int, ...]:
    """For each topology, the units of the topologies inside it in this mesh.

    The outermost topology's size enters no product, so it may be symbolic.
    Every other size MUST be a positive static int.
    """
    scales, below = [1], 1
    for topology in reversed(topologies[1:]):
        count = topology.size
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"mesh level {topology.name!r} states extent {count!r}; the index "
                "over all topologies needs the unit count of every inner topology"
            )
        below *= count
        scales.insert(0, below)
    return tuple(scales)


def scale_strides(strides, scale: int, topology, *, divide: bool = False):
    """*strides* times *scale*, or divided by it exactly when *divide* is set."""
    if isinstance(strides, tuple):
        return tuple(scale_strides(step, scale, topology, divide=divide) for step in strides)
    if scale == 1:
        return strides
    if isinstance(strides, bool):
        raise ValueError(f"mesh steps by {strides!r}, which is not a dimension")
    if not divide:
        return strides * scale if isinstance(strides, int) else services.simplify_dim(
            "mul", (strides, scale)
        )
    if isinstance(strides, int):
        exact = strides % scale == 0
    else:
        exact = services.normalize_dim(services.simplify_dim("mod", (strides, scale))) == 0
    if not exact:
        raise ValueError(
            f"mesh steps by {strides!r}, which the {scale} units inside "
            f"{topology_name(topology)!r} do not divide; its unit ids are not that level's"
        )
    if isinstance(strides, int):
        return strides // scale
    if (
        isinstance(strides, expr.Call)
        and isinstance(strides.target, expr.DimMul)
        and len(strides.args) == 2
        and expr.static_dim_value(strides.args[-1]) == scale
    ):
        return strides.args[0]
    return services.normalize_dim(services.simplify_dim("floordiv", (strides, scale)))


def layouts(mesh: Mesh) -> tuple[Layout, ...]:
    """Each topology's layout, in that topology's own numbering.

    The stored strides are in the index over all topologies; each group's
    strides are divided by the units of the topologies inside it.
    """
    stated = plain_layout(mesh.layout)
    if stated is None:
        raise ValueError(
            "a mesh whose slice states an identity box states no layout of "
            "its own, so its levels select nothing"
        )
    groups = tuple(layout.get(stated, index) for index in range(len(stated.shape)))
    if len(groups) == 1:
        return groups
    return tuple(
        Layout(group.shape, scale_strides(group.strides, scale, topology, divide=True))
        for group, scale, topology in zip(groups, unit_scales(mesh.topologies), mesh.topologies)
    )


def flat_layouts(mesh: Mesh) -> tuple[tuple[tuple, tuple], ...]:
    """Each level's layout as flat (shape, strides) int-tuples.

    A level stating no axes reads as one mode of one, so a consumer
    subscripting the levels still has something to pick.
    """
    flat = []
    for level in layouts(mesh):
        shape = tuple(flatten(level.shape))
        strides = tuple(flatten(level.strides))
        if not shape:
            shape, strides = (1,), (0,)
        flat.append((shape, strides))
    return tuple(flat)


def offset(mesh: Mesh) -> int:
    """The index of the first unit the mesh covers; 0 for an unsliced mesh."""
    if not isinstance(mesh.layout, ComposedLayout):
        return 0
    value = mesh.layout.offset
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"mesh offset {value!r} is not an integer index")
    return value


def dim_at_most(a, b) -> bool:
    """Prove ``a <= b`` from the conservative range of their difference."""
    if isinstance(a, int) and isinstance(b, int):
        return a <= b
    bounds = services.dim_range(services.simplify_dim("sub", (b, a)))
    return bounds is not None and bounds[0] >= 0


def selected_units(mesh: Mesh):
    """The unit ids a single-topology mesh selects, with its offset.

    The first item is the unit count when the ids run contiguously, else the filtered layout.
    """
    reduced = filter(plain_layout(mesh.layout), major="row")
    if not is_contiguous(reduced, major="row"):
        return reduced, offset(mesh)
    return services.normalize_dim(size(reduced)), offset(mesh)


def runs(mesh: Mesh, names) -> dict:
    """The units each named topology of *mesh* selects, for an error message."""
    return {
        topology_name(part.topologies[0]): selected_units(part)
        for part in sub_meshes(mesh)
        if topology_name(part.topologies[0]) in names
    }


def within_mesh_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether each topology of *mesh* selects a contiguous run inside the one *current* selects.

    Both runs must be contiguous; the bounds compare by the conservative range of dims.
    """
    scope = runs(current, topology_names(current))
    for part in sub_meshes(mesh):
        name = topology_name(part.topologies[0])
        if name not in scope:
            return False
        (count, start), (bound, at_start) = selected_units(part), scope[name]
        if isinstance(count, LayoutBase) or isinstance(bound, LayoutBase):
            return False
        if not dim_at_most(at_start, start) or not dim_at_most(start + count, at_start + bound):
            return False
    return True


def equals_mesh_scope(mesh: Mesh, current: Mesh) -> bool:
    """Whether each topology *mesh* names selects exactly the unit ids *current* selects for it.

    Topologies of *current* that *mesh* does not name do not enter the comparison.
    """
    scope = runs(current, topology_names(current))
    return all(
        topology_name(part.topologies[0]) in scope
        and selected_units(part) == scope[topology_name(part.topologies[0])]
        for part in sub_meshes(mesh)
    )


def check_topology(mesh: Mesh) -> None:
    """Reject static mesh units beyond their declared topology extents.

    A constant slice is already bounded by ``Mesh.__getitem__``; its shortened
    axes no longer land on full topology boundaries and are therefore accepted.
    """
    if isinstance(mesh.layout, ComposedLayout):
        return
    for topology, topology_layout in zip(mesh.topologies, layouts(mesh)):
        declared = getattr(topology, "size", None)
        if not isinstance(declared, int) or isinstance(declared, bool):
            continue
        count = size(topology_layout)
        if isinstance(count, int) and count > declared:
            raise ValueError(
                f"mesh level {topology_name(topology)!r} has {count} "
                f"units, exceeding declared extent {declared}"
            )


def joined_layout(parts: tuple[Layout, ...]) -> Layout:
    if len(parts) == 1:
        return parts[0]
    return Layout(
        shape=tuple(tuple(flatten(level.shape)) for level in parts),
        strides=tuple(tuple(flatten(level.strides)) for level in parts),
    )


def joined(
    topologies: tuple[Topology, ...],
    parts: tuple[Layout, ...],
    starts: tuple[int, ...],
    names: tuple[str, ...],
    *,
    sliced: bool,
) -> Mesh:
    scales = unit_scales(topologies)
    layout_value: Layout | ComposedLayout = joined_layout(
        tuple(
            Layout(part.shape, scale_strides(part.strides, scale, topology))
            for part, scale, topology in zip(parts, scales, topologies)
        )
    )
    if sliced:
        total = sum(start * scale for start, scale in zip(starts, scales))
        layout_value = with_offset(layout_value, total)
    return Mesh(topologies, layout_value, names)


def from_end(index: int, extent: int) -> int:
    """A Python index counted from the end, as an index from the start."""
    return index + extent if index < 0 else index


def topology_names(mesh: Mesh) -> tuple[str, ...]:
    return tuple(map(topology_name, mesh.topologies))


def axis_keys(mesh: Mesh) -> tuple[tuple[str, str | None], ...]:
    """Identify each mesh axis by its topology and authored name."""
    keys = []
    for topology, shape in zip(mesh.topologies, plain_layout(mesh.layout).shape, strict=True):
        for _ in flatten(shape):
            axis = len(keys)
            name = mesh.names[axis] if axis < len(mesh.names) else None
            keys.append((topology_name(topology), name))
    return tuple(keys)


def sub_meshes(mesh: Mesh) -> tuple[Mesh, ...]:
    """The mesh of each topology, as CuTe ``get<I>`` gives one mode at a time."""
    return tuple(get(mesh, name) for name in topology_names(mesh))


def get(value, key):
    """CuTe ``get<I>``: one mode of a layout, or the mesh of one topology of a mesh.

    For a mesh, *key* is a topology name and a name the mesh does not state gives
    ``None``. The total offset divides into that topology's own unit id here
    only. A size used to divide a nonzero remainder must be static and positive;
    once the remainder is 0 no further size is read, and the outermost keeps the quotient.
    """
    if not isinstance(value, Mesh):
        return layout.get(value, key)
    names = topology_names(value)
    if key not in names:
        return None
    if len(names) == 1:
        return value
    if isinstance(value.layout, ComposedLayout) and value.layout.inner is not None:
        raise ValueError(
            "get: a mesh of several topologies with an inner transform has no per-topology part"
        )
    at = names.index(key)
    parts = layouts(value)
    first = sum(len(flatten(part.shape)) for part in parts[:at])
    count = len(flatten(parts[at].shape))
    remaining = offset(value)
    own = 0
    for index in range(len(names) - 1, at - 1, -1):
        if index == 0:
            own = remaining
            break
        extent = value.topologies[index].size
        if remaining and (not isinstance(extent, int) or isinstance(extent, bool) or extent < 1):
            raise ValueError(
                f"get: topology {names[index]!r} extent {extent!r} cannot divide the mesh offset"
            )
        own = remaining % extent if remaining else 0
        remaining = remaining // extent if remaining else 0
    part = parts[at]
    return Mesh(
        (value.topologies[at],),
        with_offset(part, own),
        value.names[first : first + count],
    )


def sub_box(parent: Mesh, mesh: Mesh) -> tuple[slice, ...] | None:
    """Decode the first and last unit ids into a parent's sub-box."""
    if parent.topologies != mesh.topologies or not isinstance(parent.layout, Layout):
        return None
    if not isinstance(mesh.layout, ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, Layout):
        return None
    source, target = flatten(parent.layout), flatten(mesh.layout.outer)
    strides, target_strides, start = source.strides, target.strides, mesh.layout.offset
    if not all(
        isinstance(one, int)
        for one in (*source.shape, *target.shape, *strides, *target_strides, start)
    ):
        return None
    if any(step <= 0 for step in strides) or any(step < 0 for step in target_strides):
        return None
    last = start + sum((extent - 1) * step for extent, step in zip(target.shape, target_strides))
    lower = idx2crd(start, source.shape, strides)
    upper = idx2crd(last, source.shape, strides)
    if any(first > stop for first, stop in zip(lower, upper)):
        return None
    if (crd2idx(lower, source.shape, strides), crd2idx(upper, source.shape, strides)) != (start, last):
        return None
    if prod(stop - first + 1 for first, stop in zip(lower, upper)) != size(target):
        return None
    return tuple(slice(first, stop + 1) for first, stop in zip(lower, upper))
