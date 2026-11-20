"""Whether a value of one type may bind a position declared with another."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir import expr, types

from .leaves import map_tensor_types
from .local import filter_topologies
from .mesh_scope import covers_every_unit, topology_names, units_within

__all__ = ["address_layout", "canonical", "canonical_layout", "is_unsharded_layout", "types_compatible"]


def is_unsharded_layout(layout: object) -> bool:
    """Recognize the three tensor layout forms that carry no distribution."""
    shard = types.shard_layout_of(layout)
    if shard is not None:
        return all(isinstance(attr, (types.Broadcast, types.Varying)) for attr in shard.attrs)
    return isinstance(layout, (types.Layout, types.ComposedLayout))


def canonical_layout(layout):
    """One spelling per meaning: an all-Broadcast ShardLayout over every unit is its plain layout.

    See [shard §6.1](docs/spec/shard.md#61-shard-attr-semantics).
    """
    shard = types.shard_layout_of(layout)
    if shard is None or any(attr != types.Broadcast() for attr in shard.attrs) or not covers_every_unit(shard.mesh):
        return layout
    return address_layout(layout)


def address_layout(layout):
    """*layout* without its ShardLayout: the address layout one unit's copy uses."""
    shard = types.shard_layout_of(layout)
    if shard is None:
        return layout
    return shard.layout if layout is shard else replace(layout, outer=shard.layout)


def canonical(type_: types.Type) -> types.Type:
    """``canonical_layout`` on each tensor leaf of *type_*; the same object when nothing changes."""
    return map_tensor_types(
        type_,
        lambda leaf: leaf if (layout := canonical_layout(leaf.layout)) is leaf.layout else replace(leaf, layout=layout),
    )


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
        if isinstance(actual_layout, types.Layout) or (
            isinstance(actual_layout, types.ShardLayout)
            and all(attr == types.Broadcast() for attr in actual_layout.attrs)
            and units_within(declared_layout.mesh, actual_layout.mesh)
        ):
            address = actual_layout if isinstance(actual_layout, types.Layout) else actual_layout.layout
            broadcast = (types.Broadcast(),) * len(declared_layout.attrs)
            actual_layout = types.ShardLayout(address, broadcast, declared_layout.mesh)
        if not isinstance(actual_layout, types.ShardLayout):
            return False
        kept = (*topology_names(filter_topologies(declared_layout).mesh), *topology_names(filter_topologies(actual_layout).mesh))
        declared_layout = filter_topologies(declared_layout, kept)
        actual_layout = filter_topologies(actual_layout, kept)
        return (
            field_compatible(declared_layout.mesh, actual_layout.mesh)
            and len(declared_layout.attrs) == len(actual_layout.attrs)
            and all(
                held == attr or (held == types.Varying() and attr == types.Broadcast())
                for held, attr in zip(declared_layout.attrs, actual_layout.attrs)
            )
            and layout_compatible(declared_layout.layout, actual_layout.layout)
        )
    return actual_layout == declared_layout


def types_compatible(declared: types.Type, actual: types.Type) -> bool:
    """Return whether *actual* may bind a position declared as *declared*.

    Every non-``None`` declared field constrains the corresponding actual
    field. A Varying mesh axis accepts a Broadcast value; a Broadcast one
    refuses a Varying value. ``UMAT`` is the storage field's undecided value. Layout descriptors
    apply the same rule recursively.
    """
    declared, actual = canonical(declared), canonical(actual)
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
