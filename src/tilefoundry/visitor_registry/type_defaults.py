"""Materialize tensor value defaults from the target and result storage."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    LayoutBase,
    ShardLayout,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.layout import default_layout
from tilefoundry.ir.types.shard_layout import default_shard_layout, fill_shard_strides
from tilefoundry.target import instance_levels


def output_shard_layout(shape, mesh, attrs, *, ctx=None, storage=None):
    """Build a default result placement under its storage owner."""
    return default_shard_layout(
        shape, mesh, attrs, instance_levels=instance_levels(getattr(ctx, "target", None), storage)
    )


def default_type(type_, *, target=None):
    """Fill omitted value layouts without changing an authored annotation."""
    if isinstance(type_, TupleType):
        fields = tuple(default_type(t, target=target) for t in type_.fields)
        return (
            type_
            if all(a is b for a, b in zip(fields, type_.fields, strict=True))
            else replace(type_, fields=fields)
        )
    if not isinstance(type_, TensorType):
        return type_

    def layout(value, shape):
        if value is None:
            return default_layout(shape)
        if isinstance(value, Layout):
            return default_layout(value.shape) if value.strides is None else value
        if isinstance(value, ShardLayout):
            if isinstance(value.layout, Layout) and value.layout.strides is None:
                return fill_shard_strides(
                    value, instance_levels=instance_levels(target, type_.storage)
                )
            concrete = layout(value.layout, shape)
            return value if concrete is value.layout else replace(value, layout=concrete)
        if isinstance(value, ComposedLayout):
            outer = layout(value.outer, shape) if value.outer is not None else None
            inner = (
                layout(value.inner, value.inner.shape)
                if isinstance(value.inner, LayoutBase)
                else value.inner
            )
            return (
                value
                if outer is value.outer and inner is value.inner
                else replace(value, outer=outer, inner=inner)
            )
        raise TypeError(f"not a tensor layout: {value!r}")

    concrete = layout(type_.layout, type_.shape)
    return type_ if concrete is type_.layout else replace(type_, layout=concrete)
