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
    level_axes,
    shard_layout_of,
)
from .shard.layout_algebra import size
from .shard.shard_layout import split_target_axes
from .tensor_type import TensorType, TupleType, Type


def types_compatible(declared: Type, actual: Type) -> bool:
    """Return whether *actual* may bind a position declared as *declared*.

    Every non-``None`` declared field constrains the corresponding actual
    field. ``UMAT`` is the storage field's undecided value. Layout descriptors
    apply the same rule recursively.
    """
    def field_compatible(declared_field, actual_field) -> bool:
        return declared_field is None or declared_field == actual_field

    def layout_compatible(declared_layout, actual_layout) -> bool:
        if declared_layout is None:
            return True
        if isinstance(declared_layout, Layout):
            return (
                isinstance(actual_layout, Layout)
                and field_compatible(declared_layout.shape, actual_layout.shape)
                and field_compatible(declared_layout.strides, actual_layout.strides)
            )
        if isinstance(declared_layout, ShardLayout):
            return (
                isinstance(actual_layout, ShardLayout)
                and field_compatible(declared_layout.mesh, actual_layout.mesh)
                and field_compatible(declared_layout.attrs, actual_layout.attrs)
                and layout_compatible(declared_layout.layout, actual_layout.layout)
            )
        return actual_layout == declared_layout

    if isinstance(declared, TensorType):
        return (
            isinstance(actual, TensorType)
            and declared.shape == actual.shape
            and declared.dtype == actual.dtype
            and (
                declared.storage is StorageKind.UMAT
                or declared.storage == actual.storage
            )
            and layout_compatible(declared.layout, actual.layout)
        )
    if isinstance(declared, TupleType):
        return (
            isinstance(actual, TupleType)
            and len(declared.fields) == len(actual.fields)
            and all(
                types_compatible(declared_field, actual_field)
                for declared_field, actual_field in zip(declared.fields, actual.fields)
            )
        )
    return actual == declared


def numel(type: Type) -> int:
    """Element count of ``type``, summed over a tuple's leaves.

    A symbolic or negative extent is rejected rather than skipped: a size that
    silently drops a dimension reads as a smaller tensor, not as an unknown
    one. A concrete zero extent is a zero-sized tensor.
    """
    return sum(_leaf_numel(leaf) for leaf in tensor_types(type))


def _leaf_numel(type: TensorType) -> int:
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


def tensor_bytes(type: Type) -> int:
    """Byte size of ``type``, summed over a tuple's leaves.

    This is the logical size the type states, so it is the same number for
    every backend. A sub-byte dtype rounds up to whole bytes per leaf, because
    a leaf is addressed on its own.
    """
    return sum(
        math.ceil(_leaf_numel(leaf) * leaf.dtype.bit_width / 8)
        for leaf in tensor_types(type)
    )


def tensor_types(type: Type) -> tuple[TensorType, ...]:
    """The tensor leaves of *type*, flattened out of tuple nesting."""
    if isinstance(type, TensorType):
        return (type,)
    if isinstance(type, TupleType):
        return tuple(leaf for field in type.fields for leaf in tensor_types(field))
    return ()


def bytes_by_storage(
    type: Type, *, umat_level: str | None = None
) -> dict[str, int]:
    """Logical bytes occupied by *type*, grouped by storage level."""
    result: dict[str, int] = {}
    for tensor in tensor_types(type):
        if tensor.storage is StorageKind.UMAT:
            if umat_level is None:
                continue
            level = umat_level
        else:
            level = str(tensor.storage)
        result[level] = result.get(level, 0) + tensor_bytes(tensor)
    return result


def topology_extent(type: Type, name: str) -> int | None:
    """The one logical extent *type* states for topology *name*, if any."""
    extents: set[int] = set()
    for tensor in tensor_types(type):
        layout = shard_layout_of(tensor.layout)
        if layout is None:
            continue
        names = tuple(topology.name for topology in layout.mesh.topologies)
        if len(names) != 1 or names[0] != name:
            continue
        count = size(layout.mesh.layout)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(
                f"topology_extent: {name!r} needs a positive static layout size"
            )
        extents.add(count)
    if len(extents) > 1:
        raise ValueError(
            f"one value references conflicting {name!r} extents {sorted(extents)}"
        )
    return next(iter(extents), None)


def make_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind" = "gmem",
    layout: object = None,
) -> "TensorType":
    """Convenience constructor for a plain (unsharded) ``TensorType``."""
    return TensorType(shape=tuple(shape), dtype=dtype, layout=layout, storage=storage)


def make_shard_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind" = "gmem",
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


def local_type_of(
    type: Type, *, level: str | None = None, topologies: tuple[Topology, ...] = ()
) -> Type:
    """Project every tensor leaf to what one unit holds.

    With ``level``: a ``Split`` at that level or coarser divides, while finer
    splits, ``Broadcast``, and ``Partial`` do not; logical axes may factor into
    layout positions, and ``topologies`` supplies the ordered hierarchy.
    Without ``level``: every ``Split`` divides, the layout is dropped, and the
    logical rank is preserved. This form is for relations over logical axes,
    where factoring an axis into layout positions would lose the modeled flow.
    """
    if level is None:
        if not isinstance(type, TensorType):
            return type
        layout = shard_layout_of(type.layout)
        if layout is None:
            return type
        local = list(type.shape)
        for mesh_axis, tensor_axis in enumerate(split_target_axes(layout, type.shape)):
            if tensor_axis is None:
                continue
            extent = layout.mesh.layout.shape[mesh_axis]
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
                raise ValueError(
                    f"tensor axis {tensor_axis} (extent {size}) is not evenly "
                    f"divisible by its mesh extent {extent}"
                )
            local[tensor_axis] = size // extent
        return TensorType(shape=tuple(local), dtype=type.dtype, layout=None, storage=type.storage)

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
    shard = shard_layout_of(layout)
    if shard is not None:
        return TensorType(
            shape=_local_layout_shape(shard, selected_level=levels[level], topologies=topologies),
            dtype=type.dtype,
            layout=layout,
            storage=type.storage,
        )
    if isinstance(layout, (Layout, ComposedLayout)):
        return type
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
    declared = {topology.name: index for index, topology in enumerate(topologies)}
    axis_level: dict[int, int] = {}
    for topology, axes in zip(layout.mesh.topologies, level_axes(layout.mesh)):
        level = declared.get(topology.name)
        if level is None:
            raise ValueError(
                f"local_type_of: shard uses undeclared topology level {topology.name!r}"
            )
        for mesh_axis in axes:
            axis_level[mesh_axis] = level
    mesh_shape = layout.mesh.layout.shape
    for mesh_axis, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        if axis_level.get(mesh_axis, selected_level) > selected_level:
            continue
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
    _require_concrete(shape)
    return tuple(shape)
