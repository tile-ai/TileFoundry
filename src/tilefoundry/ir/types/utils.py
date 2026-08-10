from __future__ import annotations

import math
from typing import Optional

from tilefoundry.ir.types.storage import StorageKind

from .dtype import DType
from .shard import (
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Topology,
    canonical_shard_layout,
)
from .tensor_type import TensorType, TupleType, Type


def numel(type: Type) -> int:
    """Element count of ``type``, summed over a tuple's leaves.

    A symbolic or negative extent is rejected rather than skipped: a size that
    silently drops a dimension reads as a smaller tensor, not as an unknown
    one. A concrete zero extent is a zero-sized tensor.
    """
    if isinstance(type, TensorType):
        values = []
        for dim in type.shape:
            if not isinstance(dim, int) or isinstance(dim, bool):
                from .substitute import dim_vars_by_name  # noqa: PLC0415

                names = dim_vars_by_name(dim)
                hint = f"; bind it with --dim {next(iter(names))}=EXTENT" if names else ""
                raise ValueError(f"numel: tensor extent {dim!r} is not concrete{hint}")
            if dim < 0:
                raise ValueError(f"numel: tensor extent {dim} is negative")
            values.append(dim)
        return math.prod(values)
    if isinstance(type, TupleType):
        return sum(numel(field) for field in type.fields)
    return 0


def tensor_bytes(type: Type) -> int:
    """Byte size of ``type``, summed over a tuple's leaves.

    This is the logical size the type states, so it is the same number for
    every backend. A sub-byte dtype rounds up to whole bytes per leaf, because
    a leaf is addressed on its own.
    """
    if isinstance(type, TensorType):
        return math.ceil(numel(type) * type.dtype.bit_width / 8)
    if isinstance(type, TupleType):
        return sum(tensor_bytes(field) for field in type.fields)
    return 0


def make_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind | None" = "gmem",
    layout: object = None,
) -> "TensorType":
    """Convenience constructor for a plain (unsharded) ``TensorType``."""
    return TensorType(shape=tuple(shape), dtype=dtype, layout=layout, storage=storage)


def make_shard_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind | None" = "gmem",
    mesh: Optional[Mesh] = None,
    attrs: tuple = (),
) -> "TensorType":
    """Build a canonical sharded ``TensorType`` from its logical description.

    ``attrs`` contains one entry per mesh axis. With no mesh or attributes the
    result is unsharded; otherwise :func:`canonical_shard_layout` supplies the
    shared canonical representation.

    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    shape = tuple(shape)
    if mesh is None or not attrs:
        return TensorType(shape=shape, dtype=dtype, layout=None, storage=storage)
    layout = canonical_shard_layout(shape, mesh, attrs)
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


def local_type_of(type: Type, *, level: str, topologies: tuple[Topology, ...]) -> Type:
    """Project every tensor leaf to what one unit of *level* holds.

    A ``Split`` at *level* or a coarser declared level divides. Finer splits do
    not change what the containing unit holds, while ``Broadcast`` and
    ``Partial`` never divide. ``topologies`` supplies the ordered hierarchy and
    concrete extents; callers resolve launch-provided extents before projecting.
    """
    levels = {topology.name: index for index, topology in enumerate(topologies)}
    if level not in levels:
        available = ", ".join(levels) or "none"
        raise ValueError(
            f"local_type_of: topology level {level!r} is not declared; "
            f"available levels are {available}"
        )
    if len(levels) != len(topologies):
        raise ValueError("local_type_of: topology level names must be unique")
    if isinstance(type, TupleType):
        return TupleType(
            fields=tuple(
                local_type_of(field, level=level, topologies=topologies) for field in type.fields
            )
        )
    if not isinstance(type, TensorType):
        return type
    layout = type.layout
    if layout is None:
        return type
    if isinstance(layout, (Layout, ComposedLayout)):
        return type
    if isinstance(layout, ShardLayout):
        return TensorType(
            shape=_local_layout_shape(layout, selected_level=levels[level], topologies=topologies),
            dtype=type.dtype,
            layout=layout,
            storage=type.storage,
        )
    raise ValueError(
        f"local_type_of: {type!r} has unresolved layout {layout!r}; local "
        "projection requires None or a resolved ShardLayout"
    )


def _require_concrete(shape: tuple | list) -> None:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError(
            "local_type_of: local tensor extent is not a concrete non-negative integer"
        )


def _nested_layout_shape(
    layout: object, *, selected_level: int, topologies: tuple[Topology, ...]
) -> tuple:
    if isinstance(layout, ShardLayout):
        return _local_layout_shape(layout, selected_level=selected_level, topologies=topologies)
    if isinstance(layout, (Layout, ComposedLayout)):
        return tuple(layout.shape)
    raise ValueError(
        f"local_type_of: unresolved layout {layout!r}; local projection requires a resolved Layout"
    )


def _local_layout_shape(
    layout: ShardLayout, *, selected_level: int, topologies: tuple[Topology, ...]
) -> tuple[int, ...]:
    shape = list(
        _nested_layout_shape(layout.layout, selected_level=selected_level, topologies=topologies)
    )
    declared = {topology.name: (index, topology.size) for index, topology in enumerate(topologies)}
    if len(layout.mesh.topologies) > 1:
        raise ValueError(
            "local_type_of: one mesh names multiple topology levels; "
            "a level boundary cannot assign its position count"
        )
    (topology,) = layout.mesh.topologies
    resolved = declared.get(topology.name)
    if resolved is None:
        raise ValueError(f"local_type_of: shard uses undeclared topology level {topology.name!r}")
    topology_level, resolved_extent = resolved
    mesh_shape = layout.mesh.layout.shape
    launch_split_axes = tuple(
        mesh_axis
        for mesh_axis, attr in enumerate(layout.attrs)
        if isinstance(attr, Split) and mesh_axis < len(mesh_shape) and mesh_shape[mesh_axis] is None
    )
    if topology_level <= selected_level and len(launch_split_axes) > 1:
        raise ValueError(
            f"local_type_of: mesh {layout.mesh!r} has launch-provided extents "
            f"on Split axes {launch_split_axes}; topology level {topology.name!r} "
            "has one parallel width with no per-axis source, so assigning it "
            "to those axes would be a guess"
        )
    for mesh_axis, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        if topology_level > selected_level:
            continue
        if mesh_axis >= len(mesh_shape):
            raise ValueError("local_type_of: shard attribute exceeds mesh layout rank")
        mesh_extent = mesh_shape[mesh_axis]
        extent = resolved_extent if mesh_extent is None else mesh_extent
        axis = attr.axis
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < len(shape):
            raise ValueError("local_type_of: Split axis is not a concrete layout axis")
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError("local_type_of: mesh extent is not a concrete positive integer")
        if mesh_extent is None:
            if not isinstance(shape[axis], int) or isinstance(shape[axis], bool):
                raise ValueError(
                    f"local_type_of: axis {axis} has dynamic extent {shape[axis]!r}; "
                    "bind it before projecting a launch-provided mesh extent"
                )
            shape[axis] = (shape[axis] + extent - 1) // extent
        elif extent == shape[axis]:
            shape[axis] = 1
        elif not isinstance(shape[axis], int) or isinstance(shape[axis], bool):
            raise ValueError(
                f"local_type_of: axis {axis} has dynamic extent {shape[axis]!r} "
                f"and mesh extent {extent} states a fixed position count; a "
                "dynamic axis is split by a mesh sized to it (launch-provided)"
            )
        elif shape[axis] % extent:
            raise ValueError(
                f"local_type_of: extent {shape[axis]} is not divisible by mesh "
                f"extent {extent}; write the two loops out as (ceildiv(N, T), T) "
                "and bind the tile count"
            )
        else:
            shape[axis] //= extent
    _require_concrete(shape)
    return tuple(shape)
