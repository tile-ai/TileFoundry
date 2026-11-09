"""Whether a value of one type may bind a position declared with another."""

from __future__ import annotations

from tilefoundry.ir import expr, types

from .local import participant_layout

__all__ = ["is_unsharded_layout", "types_compatible"]


def is_unsharded_layout(layout: object) -> bool:
    """Recognize the three tensor layout forms that carry no distribution."""
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return all(isinstance(attr, (types.Broadcast, types.Varying)) for attr in shard.attrs)
    return isinstance(layout, (types.Layout, types.ComposedLayout))


def field_compatible(declared_field, actual_field) -> bool:
    """A field the declaration leaves ``None`` accepts any value; otherwise they must be equal."""
    return declared_field is None or declared_field == actual_field


def layout_compatible(declared_layout, actual_layout) -> bool:
    """Whether *actual_layout* may bind a position declared with *declared_layout*."""
    if declared_layout is None:
        return True
    if isinstance(declared_layout, types.Layout):
        return (
            isinstance(actual_layout, types.Layout)
            and field_compatible(declared_layout.shape, actual_layout.shape)
            and field_compatible(declared_layout.strides, actual_layout.strides)
        )
    if isinstance(declared_layout, types.ShardLayout):
        if not isinstance(actual_layout, types.ShardLayout):
            return False
        declared_layout = participant_layout(declared_layout)
        actual_layout = participant_layout(actual_layout)
        return (
            field_compatible(declared_layout.mesh, actual_layout.mesh)
            and field_compatible(declared_layout.attrs, actual_layout.attrs)
            and layout_compatible(declared_layout.layout, actual_layout.layout)
        )
    return actual_layout == declared_layout


def types_compatible(declared: types.Type, actual: types.Type) -> bool:
    """Return whether *actual* may bind a position declared as *declared*.

    Every non-``None`` declared field constrains the corresponding actual
    field. ``UMAT`` is the storage field's undecided value. Layout descriptors
    apply the same rule recursively.
    """
    if isinstance(declared, types.TensorType):
        if not isinstance(actual, types.TensorType):
            return False
        compatible_layout = declared.layout is None or layout_compatible(
            declared.layout, actual.layout
        )
        return (
            declared.shape == actual.shape
            and declared.dtype == actual.dtype
            and (declared.storage is expr.StorageKind.UMAT or declared.storage == actual.storage)
            and compatible_layout
        )
    if isinstance(declared, types.TupleType):
        return (
            isinstance(actual, types.TupleType)
            and len(declared.fields) == len(actual.fields)
            and all(
                types_compatible(declared_field, actual_field)
                for declared_field, actual_field in zip(declared.fields, actual.fields)
            )
        )
    return actual == declared
