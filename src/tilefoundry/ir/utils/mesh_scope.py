"""Which scope a program runs in, and what that scope admits.

Entering a `with Mesh(...)` composes the scope in force with the one it names;
a value is read where the units it is laid out over are the units the scope
runs on. Every function here composes CuTe functions over ``L = mesh.layout``,
the index over ``T = make_layout(sizes)``: the first unit's index is ``L(0)``
and topology ``t``'s unit ids are ``idx2crd(index, sizes)[t]``. See
[shard §5](docs/spec/shard.md#5-mesh).
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import services
from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple, layout_algebra


def flat_layout(mesh: types.Mesh) -> types.Layout | types.ComposedLayout:
    """The whole mesh as one layout, offset included: the form a mesh is written and printed in.

    The index runs across all topologies of the mesh. The reverse of ``nested_layout``.
    A sliced mesh keeps its ``ComposedLayout``, a zero offset included.
    """
    outer = types.plain_layout(mesh.layout)
    if outer is None:
        raise ValueError(
            "a mesh whose slice states an identity box states no layout of "
            "its own, so its topologies select nothing"
        )
    return types.flatten(mesh.layout)


def storage_reaches(storage, mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether *storage* reaches across a coarser value-to-scope boundary."""
    if current.topologies[-1].name in topology_names(mesh):
        return True
    try:
        storage = expr.resolve_storage(storage)
    except (TypeError, ValueError):
        return False
    return storage in {expr.StorageKind.GMEM, expr.StorageKind.SMEM}


def topology_names(mesh: types.Mesh) -> tuple[str, ...]:
    """The name of each topology of *mesh*, coarsest first."""
    return tuple(map(types.topology_name, mesh.topologies))


def axis_keys(mesh: types.Mesh) -> tuple[tuple[str, str | None], ...]:
    """Identify each mesh axis by its topology and authored name."""
    keys = []
    for topology, shape in zip(mesh.topologies, types.plain_layout(mesh.layout).shape, strict=True):
        for _ in types.flatten_to_tuple(shape):
            mesh_axis = len(keys)
            name = mesh.names[mesh_axis] if mesh_axis < len(mesh.names) else None
            keys.append((types.topology_name(topology), name))
    return tuple(keys)


def first_unit_ids(mesh: types.Mesh) -> tuple:
    """Each topology's unit id at the first unit of *mesh*: ``idx2crd(L(0), sizes)``."""
    return types.idx2crd(mesh.layout(0), types.topology_sizes(mesh.topologies))


def unit_layout(mesh: types.Mesh, t: int) -> types.Layout | types.ComposedLayout:
    """Topology *t*'s unit id at each coord of its mode.

    ``composition(U_t, get(P, t))`` with ``U_t = Layout(sizes, e_t)``, stride 1 at
    *t* and 0 elsewhere, so ``U_t(index)`` is ``idx2crd(index, sizes)[t]``. It
    starts at ``U_t(L(0))``, held as a ``ComposedLayout`` offset when it is not 0.
    A mesh naming one topology is its own: ``U_0`` is the identity.
    """
    if len(mesh.topologies) == 1:
        return types.get(mesh.layout, t)
    if isinstance(mesh.layout, types.ComposedLayout) and mesh.layout.inner is not None:
        raise ValueError(
            "a mesh of several topologies with an inner transform has no per-topology unit ids"
        )
    sizes = types.topology_sizes(mesh.topologies)
    units = types.Layout(sizes, tuple(int(at == t) for at in range(len(sizes))))
    mode = types.get(types.plain_layout(mesh.layout), t)
    try:
        ids = layout_algebra.composition(units, mode)
    except TypeError as error:
        raise ValueError(
            f"topology {types.topology_name(mesh.topologies[t])!r}: composition cannot "
            f"order the symbolic strides of its mode {mode}"
        ) from error
    first_unit_id = units(mesh.layout(0))
    return types.make_composed_layout(None, first_unit_id, ids) if first_unit_id else ids


def topology_mesh(mesh: types.Mesh, name: str) -> types.Mesh | None:
    """Topology *name* of *mesh* as a mesh of that topology alone, in its unit ids.

    Its layout is ``unit_layout``; ``None`` when *mesh* does not name *name*.
    """
    names = topology_names(mesh)
    if name not in names:
        return None
    if len(names) == 1:
        return mesh
    t = names.index(name)
    stated = types.plain_layout(mesh.layout)
    first = sum(types.rank(types.flatten(types.get(stated, at))) for at in range(t))
    count = types.rank(types.flatten(types.get(stated, t)))
    return types.Mesh((mesh.topologies[t],), unit_layout(mesh, t), mesh.names[first : first + count])


def selected_units(mesh: types.Mesh, t: int):
    """The unit ids topology *t* of *mesh* selects, with the first of them.

    The first item is the unit count when the ids run contiguously, else the
    filtered ``unit_layout``.
    """
    reduced = layout_algebra.filter(types.plain_layout(unit_layout(mesh, t)))
    first_unit_id = first_unit_ids(mesh)[t]
    if not layout_algebra.is_contiguous(reduced):
        return reduced, first_unit_id
    return services.normalize_dim(types.size(reduced)), first_unit_id


def runs(mesh: types.Mesh, names) -> dict:
    """The units each named topology of *mesh* selects, for an error message."""
    return {
        name: selected_units(mesh, t)
        for t, name in enumerate(topology_names(mesh))
        if name in names
    }


def within_mesh_scope(mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether each topology of *mesh* selects a contiguous run inside the one *current* selects.

    Both runs must be contiguous; the bounds compare by the conservative range of dims.
    """
    scope = runs(current, topology_names(current))
    for t, name in enumerate(topology_names(mesh)):
        if name not in scope:
            return False
        (count, first_unit_id), (bound, scope_first_unit_id) = selected_units(mesh, t), scope[name]
        if isinstance(count, types.LayoutBase) or isinstance(bound, types.LayoutBase):
            return False
        if not int_tuple.dim_at_most(scope_first_unit_id, first_unit_id) or not int_tuple.dim_at_most(
            first_unit_id + count, scope_first_unit_id + bound
        ):
            return False
    return True


def equals_mesh_scope(mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether each topology *mesh* names selects exactly the unit ids *current* selects for it.

    Topologies of *current* that *mesh* does not name do not enter the comparison.
    """
    scope = runs(current, topology_names(current))
    return all(
        name in scope and selected_units(mesh, t) == scope[name]
        for t, name in enumerate(topology_names(mesh))
    )


@dataclass(frozen=True)
class UnitChange:
    """How the units of two meshes relate, over the topologies both name.

    ``topology`` is the coarsest of them whose selection differs; the counts are
    the units each mesh selects and the units both select.
    """

    topology: str
    source: int
    destination: int
    shared: int

    @property
    def added(self) -> int:
        """Units the destination holds and the source does not."""
        return self.destination - self.shared

    @property
    def overlap_in_part(self) -> bool:
        """Whether the two share some units; their selections differ, so not all of them."""
        return self.shared > 0


def unit_ids(run) -> frozenset[int] | None:
    """The exact unit ids a ``selected_units`` run names; None when they are symbolic."""
    held, first = run
    if isinstance(held, types.LayoutBase):
        count = types.size(held)
        ids = tuple(first + held(index) for index in range(count)) if isinstance(count, int) else ()
    else:
        ids = tuple(range(first, first + held)) if isinstance(held, int) and isinstance(first, int) else ()
    return frozenset(ids) if ids and all(isinstance(unit, int) for unit in ids) else None


def unit_change(source: types.Mesh, destination: types.Mesh, topologies) -> UnitChange | None:
    """Compare the exact unit sets two meshes select, on the named *topologies*.

    See [shard §5](docs/spec/shard.md#5-mesh). None when they select the same
    units there or name none of them in common.

    Raises:
        ValueError: The selections differ and a symbolic run cannot be counted.
    """
    before = runs(source, topology_names(source))
    after = runs(destination, topology_names(destination))
    common = [name for name in after if name in before and name in topologies]
    counts, changed = {"source": 1, "destination": 1, "shared": 1}, None
    for name in common:
        if before[name] == after[name]:
            held = unit_ids(before[name])
            if held is not None:
                counts["source"] *= len(held)
                counts["destination"] *= len(held)
                counts["shared"] *= len(held)
            continue
        held, wanted = unit_ids(before[name]), unit_ids(after[name])
        if held is None or wanted is None:
            raise ValueError(
                f"the {name} units {before[name]} and {after[name]} differ, and a symbolic "
                "selection cannot say how many units gain the value"
            )
        counts["source"] *= len(held)
        counts["destination"] *= len(wanted)
        counts["shared"] *= len(held & wanted)
        if changed is None and held != wanted:
            changed = name
    return None if changed is None else UnitChange(changed, **counts)


def from_meshes(*meshes: types.Mesh) -> types.Mesh:
    """Compose nested mesh scopes, preserving slices in the index over all topologies.

    A scope naming topologies disjoint from the one in force adds them inside it;
    one naming a suffix of its topologies replaces that suffix and must select
    within it; one naming all of them replaces it.
    """
    if not meshes:
        raise ValueError("make_mesh requires at least one mesh")
    result = meshes[0]
    for inner in meshes[1:]:
        here, there = topology_names(result), topology_names(inner)
        if set(here).isdisjoint(there):
            result = joined(result, len(here), inner, (*result.names, *inner.names))
        elif set(here) <= set(there):
            result = inner
        elif len(there) < len(here) and here[-len(there) :] == there:
            current = result
            kept = len(here) - len(there)
            stated = types.plain_layout(result.layout)
            named = sum(types.rank(types.flatten(types.get(stated, t))) for t in range(kept))
            result = joined(result, kept, inner, (*result.names[:named], *inner.names))
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
                "replaces the topologies in force or adds topologies below them"
            )
    types.check_topology(result)
    return result


def joined(outer: types.Mesh, kept: int, inner: types.Mesh, names: tuple) -> types.Mesh:
    """The first *kept* topologies of *outer* with every topology of *inner* inside them.

    Each topology's ``unit_layout`` moves to the new ``T = make_layout(sizes)`` by
    ``composition(get(T, t), ·)``; the first index is ``crd2idx`` of the first unit
    ids in ``T``.
    """
    topologies = (*outer.topologies[:kept], *inner.topologies)
    if any(not isinstance(topology, types.Topology) for topology in topologies):
        raise ValueError("a multi-topology Mesh requires Topology values")
    sizes = types.topology_sizes(topologies)
    units = types.compact_row_major(sizes)
    modes, ids = [], []
    for mesh, count in ((outer, kept), (inner, len(inner.topologies))):
        for t in range(count):
            at = len(modes)
            mode = types.plain_layout(unit_layout(mesh, t))
            modes.append(layout_algebra.composition(types.Layout(sizes[at], units[at]), mode))
        ids.extend(first_unit_ids(mesh)[:count])
    stated = types.make_layout(*modes)
    first = types.crd2idx(tuple(ids), sizes, units)
    sliced = isinstance(outer.layout, types.ComposedLayout) or isinstance(
        inner.layout, types.ComposedLayout
    )
    return types.Mesh(
        topologies,
        types.make_composed_layout(None, first, stated) if sliced and first else stated,
        names,
    )


def sub_box(parent: types.Mesh, mesh: types.Mesh) -> tuple[slice, ...] | None:
    """The box of *parent* whose slice is *mesh*, from the coords of its first and last units.

    The first and last indices ``L(0)`` and ``L(size - 1)`` read back as parent
    coords by ``idx2crd``; the box is refused unless ``crd2idx`` returns them and its
    size is the selected size.
    """
    if parent.topologies != mesh.topologies or not isinstance(parent.layout, types.Layout):
        return None
    if not isinstance(mesh.layout, types.ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, types.Layout):
        return None
    source, target = types.flatten(parent.layout), types.flatten(mesh.layout.outer)
    strides, target_strides = source.strides, target.strides
    if not all(
        isinstance(one, int)
        for one in (*source.shape, *target.shape, *strides, *target_strides, mesh.layout.offset)
    ):
        return None
    if any(step <= 0 for extent, step in zip(source.shape, strides) if extent != 1) or any(
        step < 0 for step in target_strides
    ):
        return None
    first_index, last = mesh.layout(0), mesh.layout(types.size(target) - 1)
    lower = types.idx2crd(first_index, source.shape, strides)
    upper = types.idx2crd(last, source.shape, strides)
    if any(first > stop for first, stop in zip(lower, upper)):
        return None
    if (
        types.crd2idx(lower, source.shape, strides),
        types.crd2idx(upper, source.shape, strides),
    ) != (first_index, last):
        return None
    if int_tuple.product(tuple(stop - first + 1 for first, stop in zip(lower, upper))) != types.size(
        target
    ):
        return None
    return tuple(slice(first, stop + 1) for first, stop in zip(lower, upper))


__all__ = [
    "axis_keys",
    "equals_mesh_scope",
    "first_unit_ids",
    "flat_layout",
    "from_meshes",
    "runs",
    "selected_units",
    "storage_reaches",
    "sub_box",
    "topology_mesh",
    "topology_names",
    "unit_change",
    "unit_ids",
    "unit_layout",
    "within_mesh_scope",
    "UnitChange",
]
