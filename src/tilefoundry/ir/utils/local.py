"""The local view of a type: what one unit of a topology holds."""

from __future__ import annotations

from typing import Optional

from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple

from .mesh_scope import axis_keys, first_unit_ids, from_meshes, topology_mesh, topology_names

__all__ = [
    "divided_axes",
    "filter_topologies",
    "find_execution_mesh",
    "local_type_of",
    "make_shard_tensor_type",
    "physical_layout",
    "topology_level",
]


def filter_topologies(
    layout: types.ShardLayout, required: types.Mesh | tuple[str, ...] | None = None
) -> types.ShardLayout:
    """Drop the topologies a value does not take part on, keeping mesh axes and attrs aligned.

    CuTe ``filter`` drops modes that do not change the index. An instruction
    retains all axes of its required topologies, including Broadcast. Outside
    them, only Split and Partial prevent omission. Without an instruction's
    execution mesh, retain value-varying topologies too, or the innermost
    topology for an entirely broadcast value.
    """
    names = topology_names(layout.mesh)
    required_names = (
        set(topology_names(required))
        if isinstance(required, types.Mesh)
        else None
        if required is None
        else set(required)
    )
    if len(names) == 1 and (required_names is None or names[0] in required_names):
        return layout
    keys = axis_keys(layout.mesh)
    if len(keys) != len(layout.attrs):
        raise ValueError("shard attributes must have the rank of their mesh")
    attrs_by_topology = {name: [] for name in names}
    for attr, (name, _) in zip(layout.attrs, keys):
        attrs_by_topology[name].append(attr)
    if required_names is None:
        required_names = {
            name
            for name, attrs in attrs_by_topology.items()
            if any(not isinstance(attr, types.Broadcast) for attr in attrs)
        }
    if not required_names:
        required_names = {names[-1]}
    selected = []
    selected_attrs = []
    for name, attrs in attrs_by_topology.items():
        if name in required_names:
            selected.append(topology_mesh(layout.mesh, name))
            selected_attrs.extend(attrs)
        elif any(isinstance(attr, (types.Split, types.Partial)) for attr in attrs):
            raise ValueError(f"shard attr other than Broadcast outside the kept topology {name!r}")
    if not selected:
        raise ValueError("shard layout keeps no topology")
    if len(selected) == len(names):
        return layout
    return types.ShardLayout(layout.layout, tuple(selected_attrs), from_meshes(*selected))


def physical_layout(layout: types.ShardLayout) -> types.LayoutBase:
    """Omit value-only levels, then read remaining variation as Broadcast."""
    if not any(isinstance(attr, types.Varying) for attr in layout.attrs):
        return layout
    by_level = {level: [] for level, _ in axis_keys(layout.mesh)}
    for attr, (level, _) in zip(layout.attrs, axis_keys(layout.mesh)):
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
    selected = filter_topologies(layout, kept)
    return types.ShardLayout(
        selected.layout, tuple(map(types.physical_attr, selected.attrs)), selected.mesh
    )


def find_execution_mesh(mesh: types.Mesh, required: types.Mesh) -> tuple[types.Mesh, int]:
    """The mesh axes at the end of *mesh* that cover exactly the units of an op's execution mesh.

    *required* is the execution mesh the op declares. The int is how many mesh
    axes come before those axes, as an index into the full layout of *mesh*.
    """
    assert len(required.topologies) == 1, "an execution mesh has one topology"
    source_layout = types.flatten(types.plain_layout(mesh.layout))
    required_layout = types.flatten(types.plain_layout(required.layout))
    if not isinstance(source_layout, types.Layout) or not isinstance(required_layout, types.Layout):
        raise ValueError("tile execution meshes require concrete layouts")
    source_shape = tuple(source_layout.shape)
    source_strides = tuple(source_layout.strides)
    stated = {required_layout(index) for index in range(types.size(required_layout))}
    dropped = next(
        (
            begin
            for begin in range(len(source_shape) - 1, -1, -1)
            if (
                (suffix := types.Layout(source_shape[begin:], source_strides[begin:]))
                and {suffix(index) for index in range(types.size(suffix))} == stated
            )
        ),
        None,
    )
    if dropped is None:
        raise ValueError(
            f"mesh layout {source_shape} does not end in execution mesh "
            f"{tuple(required_layout.shape)}"
        )
    start = first_unit_ids(mesh)[-1]
    return (
        types.Mesh(
            mesh.topologies[-len(required.topologies) :],
            types.make_composed_layout(None, start, required.layout) if start else required.layout,
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
        local[tensor_axis] = int_tuple.shape_div(size, extent)
    return types.TensorType(
        shape=tuple(local),
        dtype=type.dtype,
        layout=types.Layout(tuple(local)),
        storage=type.storage,
    )


def local_type_of(
    type: types.Type, *, topology: str | None = None, topologies: tuple[types.Topology, ...] = ()
) -> types.Type:
    """Project every tensor leaf to what one unit holds.

    With ``topology``: a ``Split`` at that topology or coarser divides, while
    finer splits, ``Broadcast``, and ``Partial`` do not; logical axes may factor
    into layout modes, and ``topologies`` supplies the ordered hierarchy.
    Without it: every ``Split`` divides, the layout is dropped, and the logical
    rank is preserved. This form is for relations over logical axes, where
    factoring an axis into layout modes would lose the modeled flow.
    """
    if topology is None:
        projected = all_split_local_type(type, refuse_indivisible=True)
        assert projected is not None
        return projected

    selected = topology_level(topology, topologies)
    if isinstance(type, types.TupleType):
        return types.TupleType(
            fields=tuple(
                local_type_of(field, topology=topology, topologies=topologies)
                for field in type.fields
            )
        )
    if not isinstance(type, types.TensorType):
        return type
    layout = type.layout
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return types.TensorType(
            shape=local_layout_shape(shard, selected_level=selected, topologies=topologies),
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


def topology_level(topology: str, topologies: tuple[types.Topology, ...]) -> int:
    """Where *topology* sits in the Module's declared *topologies*, coarsest first."""
    declared = {each.name: index for index, each in enumerate(topologies)}
    if topology not in declared:
        available = ", ".join(declared) or "none"
        raise ValueError(
            f"local_type_of: topology {topology!r} is not declared; "
            f"available topologies are {available}"
        )
    if len(declared) != len(topologies):
        raise ValueError("local_type_of: topology names must be unique")
    return declared[topology]


def divided_axes(
    layout: types.ShardLayout, *, selected_level: int, topologies: tuple[types.Topology, ...]
) -> tuple[int, ...]:
    """The mesh axes whose ``Split`` one unit at topology level *selected_level* holds one part of.

    A ``Split`` at that level or coarser divides; a finer one does not. This is
    the rule ``local_type_of`` divides by, stated once for whoever places a
    value's coordinates among that unit's modes.
    """
    axis_topology_levels = tuple(
        topology_level(name, topologies) for name, _ in axis_keys(layout.mesh)
    )
    return tuple(
        mesh_axis
        for mesh_axis, attr in enumerate(layout.attrs)
        if isinstance(attr, types.Split)
        and (
            axis_topology_levels[mesh_axis]
            if mesh_axis < len(axis_topology_levels)
            else selected_level
        )
        <= selected_level
    )


def require_concrete(shape: tuple | list) -> None:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError(
            "local_type_of: local tensor extent is not a concrete non-negative integer"
        )


def nested_layout_shape(
    layout: object, *, selected_level: int, topologies: tuple[types.Topology, ...]
) -> tuple:
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return local_layout_shape(
            shard, selected_level=selected_level, topologies=topologies
        )
    if isinstance(layout, (types.Layout, types.ComposedLayout)):
        return types.flatten_to_tuple(layout.shape)
    raise ValueError(
        f"local_type_of: unresolved layout {layout!r}; local projection requires a resolved Layout"
    )


def local_layout_shape(
    layout: types.ShardLayout, *, selected_level: int, topologies: tuple[types.Topology, ...]
) -> tuple[int, ...]:
    shape = list(
        nested_layout_shape(
            layout.layout, selected_level=selected_level, topologies=topologies
        )
    )
    mesh_shape = types.flatten(layout.mesh.layout).shape
    for mesh_axis in divided_axes(
        layout, selected_level=selected_level, topologies=topologies
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
                f"and mesh extent {extent} states a fixed unit count; bind "
                "the axis before local projection"
            )
        elif shape[axis] % extent:
            raise ValueError(
                f"local_type_of: extent {shape[axis]} is not divisible by mesh "
                f"extent {extent}; write the two loops out as (ceildiv(N, T), T) "
                "and bind the tile count"
            )
        else:
            shape[axis] = int_tuple.shape_div(shape[axis], extent)
    require_concrete(shape)
    return tuple(shape)
