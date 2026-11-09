"""The local view of a type: what one unit of a topology level holds."""

from __future__ import annotations

from typing import Optional

from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple

__all__ = [
    "divided_mesh_axes",
    "local_type_of",
    "make_shard_tensor_type",
    "participant_frame",
    "participant_layout",
    "physical_layout",
    "try_local_type_of",
]


def topology_label(topology):
    """The name of a topology, or the string itself when one was written as a string."""
    return getattr(topology, "name", topology)


def participant_layout(
    layout: types.ShardLayout, required: types.Mesh | tuple[str, ...] | None = None
) -> types.ShardLayout:
    """Select participating levels while keeping their mesh axes and attrs aligned.

    An instruction retains all axes of its required levels, including Broadcast.
    Outside an instruction's required levels, only Split and Partial prevent
    omission. Without an instruction frame, retain value-varying levels too,
    or the innermost level for an entirely broadcast value.
    """
    levels = types.separate(layout.mesh)
    required_names = (
        {topology_label(topology) for topology in required.topologies}
        if isinstance(required, types.Mesh)
        else None
        if required is None
        else set(required)
    )
    if len(levels) == 1 and (
        required_names is None or topology_label(levels[0].topologies[0]) in required_names
    ):
        return layout
    attrs_by_level = []
    offset = 0
    for level in levels:
        rank = len(types.flatten(level.layout).shape)
        attrs_by_level.append(layout.attrs[offset : offset + rank])
        offset += rank
    if offset != len(layout.attrs):
        raise ValueError("shard attributes must have the rank of their mesh")
    if required_names is None:
        required_names = {
            topology_label(level.topologies[0])
            for level, attrs in zip(levels, attrs_by_level, strict=True)
            if any(not isinstance(attr, types.Broadcast) for attr in attrs)
        }
    if not required_names:
        required_names = {topology_label(levels[-1].topologies[0])}
    selected = []
    selected_attrs = []
    for level, attrs in zip(levels, attrs_by_level, strict=True):
        name = topology_label(level.topologies[0])
        if name in required_names:
            selected.append(level)
            selected_attrs.extend(attrs)
        elif any(isinstance(attr, (types.Split, types.Partial)) for attr in attrs):
            raise ValueError(f"non-Broadcast placement outside participant level {name!r}")
    if not selected:
        raise ValueError("shard layout has no participant level")
    if len(selected) == len(levels):
        return layout
    return types.ShardLayout(layout.layout, tuple(selected_attrs), types.Mesh.from_meshes(*selected))


def physical_layout(layout: types.ShardLayout) -> types.LayoutBase:
    """Omit value-only levels, then read remaining variation as Broadcast."""
    if not any(isinstance(attr, types.Varying) for attr in layout.attrs):
        return layout
    by_level = {level: [] for level, _ in types.axis_keys(layout.mesh)}
    for attr, (level, _) in zip(layout.attrs, types.axis_keys(layout.mesh)):
        by_level[level].append(attr)
    kept = tuple(
        level
        for level, attrs in by_level.items()
        if not (
            any(isinstance(attr, types.Varying) for attr in attrs)
            and not any(isinstance(attr, (types.Split, types.Partial)) for attr in attrs)
        )
    )
    if not kept:
        return layout.layout
    selected = participant_layout(layout, kept)
    return types.ShardLayout(
        selected.layout, tuple(map(types.physical_attr, selected.attrs)), selected.mesh
    )


def participant_frame(source: types.Mesh, required: types.Mesh) -> tuple[types.Mesh, int]:
    """Find a physical suffix; its dropped count indexes the full source layout."""
    assert len(required.topologies) == 1, "participant frames require one topology level"
    source_layout = types.flatten(source.layout)
    required_layout = types.flatten(required.layout)
    if not isinstance(source_layout, types.Layout) or not isinstance(required_layout, types.Layout):
        raise ValueError("tile participant meshes require concrete layouts")
    source_shape = tuple(source_layout.shape)
    source_strides = tuple(source_layout.strides)
    stated = {types.apply(required_layout, index) for index in range(types.size(required_layout))}
    dropped = next(
        (
            begin
            for begin in range(len(source_shape) - 1, -1, -1)
            if (
                (suffix := types.Layout(source_shape[begin:], source_strides[begin:]))
                and {types.apply(suffix, index) for index in range(types.size(suffix))} == stated
            )
        ),
        None,
    )
    if dropped is None:
        raise ValueError(
            f"mesh layout {source_shape} does not end in participant frame "
            f"{tuple(required_layout.shape)}"
        )
    return (
        types.Mesh(
            source.topologies[-len(required.topologies) :],
            required.layout
            if types.starts(source)[-1] == 0
            else types.ComposedLayout(None, types.starts(source)[-1], required.layout),
            required.names,
        ),
        dropped,
    )


def make_shard_tensor_type(
    shape: tuple,
    dtype: expr.DType = expr.DType.f32,
    storage: "str | expr.StorageKind" = "gmem",
    mesh: Optional[types.Mesh] = None,
    attrs: tuple = (),
) -> "types.TensorType":
    """Build a canonical sharded ``TensorType`` from its logical description.

    ``attrs`` contains one entry per mesh axis. With no mesh or attributes the
    result is unsharded; otherwise ``ShardLayout.from_logical`` supplies the
    shared canonical representation.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    shape = tuple(shape)
    if mesh is None or not attrs:
        return types.TensorType(shape=shape, dtype=dtype, layout=types.Layout(shape), storage=storage)
    layout = types.ShardLayout.from_logical(shape, mesh, attrs, storage=storage, owners=expr.StorageOwners((), {}))
    return types.TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


def all_split_local_type(type: types.Type, *, refuse_indivisible: bool) -> types.Type | None:
    """Project every Split, optionally returning None for an indivisible axis."""
    if not isinstance(type, types.TensorType):
        return type
    layout = types.shard_layout_of(type.layout)
    if layout is None:
        return type
    local = list(type.shape)
    for mesh_axis, tensor_axis in enumerate(types.split_target_axes(layout, type.shape)):
        if tensor_axis is None:
            continue
        extent = types.flatten(layout.mesh.layout).shape[mesh_axis]
        if extent is None:
            local[tensor_axis] = 1
            continue
        size = local[tensor_axis]
        if not isinstance(size, int) or isinstance(size, bool):
            raise ValueError(
                f"tensor axis {tensor_axis} is Split-sharded but its extent "
                f"{size!r} is not a static int"
            )
        if size % extent != 0:
            if not refuse_indivisible:
                return None
            raise ValueError(
                f"tensor axis {tensor_axis} (extent {size}) is not evenly "
                f"divisible by its mesh extent {extent}"
            )
        local[tensor_axis] = size // extent
    return types.TensorType(
        shape=tuple(local),
        dtype=type.dtype,
        layout=types.Layout(tuple(local)),
        storage=type.storage,
    )


def try_local_type_of(type: types.Type) -> types.Type | None:
    """Project logical splits, returning None when a factored split has no local box.

    Sequential division along logical axes can fail for factored or reordered
    shard layouts (for example, 16 / 4 followed by 4 % 8, or 16 % 32).
    Candidates compare the whole type when that logical projection has no
    solution; this does not decide whether an instruction issues whole.
    """
    return all_split_local_type(type, refuse_indivisible=False)


def local_type_of(
    type: types.Type, *, topology_level: str | None = None, topologies: tuple[types.Topology, ...] = ()
) -> types.Type:
    """Project every tensor leaf to what one unit holds.

    With ``topology_level``: a ``Split`` at that level or coarser divides, while
    finer splits, ``Broadcast``, and ``Partial`` do not; logical axes may factor
    into layout positions, and ``topologies`` supplies the ordered hierarchy.
    Without it: every ``Split`` divides, the layout is dropped, and the logical
    rank is preserved. This form is for relations over logical axes, where
    factoring an axis into layout positions would lose the modeled flow.
    """
    if topology_level is None:
        projected = all_split_local_type(type, refuse_indivisible=True)
        assert projected is not None
        return projected

    selected = level_index(topology_level, topologies)
    if isinstance(type, types.TupleType):
        return types.TupleType(
            fields=tuple(
                local_type_of(field, topology_level=topology_level, topologies=topologies)
                for field in type.fields
            )
        )
    if not isinstance(type, types.TensorType):
        return type
    layout = type.layout
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return types.TensorType(
            shape=local_layout_shape(
                shard, selected_topology_level=selected, topologies=topologies
            ),
            dtype=type.dtype,
            layout=layout,
            storage=type.storage,
        )
    if isinstance(layout, (types.Layout, types.ComposedLayout)):
        return type
    raise ValueError(
        f"local_type_of: {type!r} has unresolved layout {layout!r}; local "
        "projection requires a concrete layout"
    )


def level_index(topology_level: str, topologies: tuple[types.Topology, ...]) -> int:
    """Where *topology_level* sits in the declared hierarchy, coarsest first."""
    levels = {topology.name: index for index, topology in enumerate(topologies)}
    if topology_level not in levels:
        available = ", ".join(levels) or "none"
        raise ValueError(
            f"local_type_of: topology level {topology_level!r} is not declared; "
            f"available levels are {available}"
        )
    if len(levels) != len(topologies):
        raise ValueError("local_type_of: topology level names must be unique")
    return levels[topology_level]


def divided_mesh_axes(
    layout: types.ShardLayout, *, topology_level: str, topologies: tuple[types.Topology, ...]
) -> tuple[int, ...]:
    """The mesh axes whose ``Split`` one unit of *topology_level* holds one part of.

    A ``Split`` at that level or coarser divides; a finer one does not. This is
    the rule ``local_type_of`` divides by, stated once for whoever places a
    value's coordinates among that unit's positions.
    """
    selected = level_index(topology_level, topologies)
    return divided_axes(layout, selected_topology_level=selected, topologies=topologies)


def divided_axes(
    layout: types.ShardLayout, *, selected_topology_level: int, topologies: tuple[types.Topology, ...]
) -> tuple[int, ...]:
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    for topology in layout.mesh.topologies:
        if topology.name not in declared:
            raise ValueError(
                f"local_type_of: shard uses undeclared topology level {topology.name!r}"
            )
    mesh_layout = layout.mesh.layout
    stated = mesh_layout.outer if isinstance(mesh_layout, types.ComposedLayout) else mesh_layout
    axis_topology_level = types.flatten(
        tuple(
            int_tuple.repeat_like(mode, declared[topology.name])
            for mode, topology in zip(stated.shape, layout.mesh.topologies, strict=True)
        )
    )
    return tuple(
        mesh_axis
        for mesh_axis, attr in enumerate(layout.attrs)
        if isinstance(attr, types.Split)
        and (
            axis_topology_level[mesh_axis]
            if mesh_axis < len(axis_topology_level)
            else selected_topology_level
        )
        <= selected_topology_level
    )


def require_concrete(shape: tuple | list) -> None:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError(
            "local_type_of: local tensor extent is not a concrete non-negative integer"
        )


def nested_layout_shape(
    layout: object, *, selected_topology_level: int, topologies: tuple[types.Topology, ...]
) -> tuple:
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return local_layout_shape(
            shard, selected_topology_level=selected_topology_level, topologies=topologies
        )
    if isinstance(layout, (types.Layout, types.ComposedLayout)):
        return tuple(types.flatten(layout).shape)
    raise ValueError(
        f"local_type_of: unresolved layout {layout!r}; local projection requires a resolved Layout"
    )


def local_layout_shape(
    layout: types.ShardLayout, *, selected_topology_level: int, topologies: tuple[types.Topology, ...]
) -> tuple[int, ...]:
    shape = list(
        nested_layout_shape(
            layout.layout, selected_topology_level=selected_topology_level, topologies=topologies
        )
    )
    mesh_shape = types.flatten(layout.mesh.layout).shape
    for mesh_axis in divided_axes(
        layout, selected_topology_level=selected_topology_level, topologies=topologies
    ):
        attr = layout.attrs[mesh_axis]
        if mesh_axis >= len(mesh_shape):
            raise ValueError("local_type_of: shard attribute exceeds mesh layout rank")
        extent = mesh_shape[mesh_axis]
        axis = attr.axis
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < len(shape):
            raise ValueError("local_type_of: Split axis is not a concrete layout axis")
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError("local_type_of: mesh extent is not a concrete positive integer")
        if extent == shape[axis]:
            shape[axis] = 1
        elif not isinstance(shape[axis], int) or isinstance(shape[axis], bool):
            raise ValueError(
                f"local_type_of: axis {axis} has dynamic extent {shape[axis]!r} "
                f"and mesh extent {extent} states a fixed position count; bind "
                "the axis before local projection"
            )
        elif shape[axis] % extent:
            raise ValueError(
                f"local_type_of: extent {shape[axis]} is not divisible by mesh "
                f"extent {extent}; write the two loops out as (ceildiv(N, T), T) "
                "and bind the tile count"
            )
        else:
            shape[axis] //= extent
    require_concrete(shape)
    return tuple(shape)
