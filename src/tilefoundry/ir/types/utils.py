from __future__ import annotations

import math
from typing import Optional

from tilefoundry.ir.types.storage import StorageKind

from .dtype import DType
from .shard import ComposedLayout, Layout, Mesh, ShardLayout, Split, canonical_shard_layout
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
                hint = (
                    f"; bind it with --dim {next(iter(names))}=EXTENT" if names else ""
                )
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
    """Build the canonical sharded ``TensorType`` (``docs/spec/shard.md``
    §7.1.1) from a logical description: ``shape`` is the logical tensor
    shape, ``attrs`` is one entry per mesh axis (``Split(logical_axis)`` /
    ``Broadcast()`` / ``Partial(reduction)``). ``mesh=None`` / ``attrs=()``
    yields a plain (unsharded) ``TensorType``; otherwise the layout is built
    by the shared :func:`canonical_shard_layout` (also used by
    ``derive_output_shard_layout``'s synthesis fallback, so the two
    producers of a §7.1.1 layout always agree).
    """
    shape = tuple(shape)
    if mesh is None or not attrs:
        return TensorType(shape=shape, dtype=dtype, layout=None, storage=storage)
    layout = canonical_shard_layout(shape, mesh, attrs)
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


def local_type_of(type: Type) -> Type:
    """Project every ``TensorType`` leaf of ``type`` to its per-shard local
    shape, rebuilding ``TupleType`` structure; any other ``Type`` passes
    through unchanged.

    Applies every already-resolved nested ``ShardLayout`` exactly once. A
    ``TensorType`` whose layout is neither ``None`` nor a resolved layout is
    rejected — the caller must resolve it before requesting a local projection.
    """
    if isinstance(type, TupleType):
        return TupleType(fields=tuple(local_type_of(field) for field in type.fields))
    if not isinstance(type, TensorType):
        return type
    layout = type.layout
    if layout is None:
        return type
    if isinstance(layout, (Layout, ComposedLayout)):
        return type
    if isinstance(layout, ShardLayout):
        return TensorType(
            shape=_local_layout_shape(layout),
            dtype=type.dtype,
            layout=layout,
            storage=type.storage,
        )
    raise ValueError(
        f"local_type_of: {type!r} has unresolved layout {layout!r}; local "
        "projection requires None or a resolved ShardLayout"
    )


def _layout_shape(layout: object) -> tuple:
    if isinstance(layout, ShardLayout):
        return _local_layout_shape(layout)
    if isinstance(layout, (Layout, ComposedLayout)):
        shape = tuple(layout.shape)
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError(
                "local_type_of: local tensor extent is not a concrete non-negative integer"
            )
        return shape
    raise ValueError(
        f"local_type_of: unresolved layout {layout!r}; local projection requires "
        "a resolved Layout"
    )


def _local_layout_shape(layout: ShardLayout) -> tuple[int, ...]:
    shape = list(_layout_shape(layout.layout))
    for mesh_axis, attr in enumerate(layout.attrs):
        if not isinstance(attr, Split):
            continue
        if mesh_axis >= len(layout.mesh.layout.shape):
            raise ValueError("local_type_of: shard attribute exceeds mesh rank")
        axis = attr.axis
        extent = layout.mesh.layout.shape[mesh_axis]
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < len(shape):
            raise ValueError("local_type_of: Split axis is not a concrete layout axis")
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError("local_type_of: mesh extent is not a concrete positive integer")
        if shape[axis] % extent:
            raise ValueError(
                f"local_type_of: extent {shape[axis]} is not divisible by mesh extent {extent}"
            )
        shape[axis] //= extent
    return tuple(shape)
