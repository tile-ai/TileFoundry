"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

import itertools
from collections.abc import Callable

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.ir.types.utils import static_dim_value

from .registries import DispatchRegistry

buffer_alias_registry: DispatchRegistry = DispatchRegistry("buffer_alias")


def register_buffer_alias(
    op_cls: type, param: ParamDef, when: "Callable[[Call], bool] | None" = None
) -> None:
    """Declare that op_cls's result re-addresses its ``param`` operand's bytes.

    ``when`` narrows that to the calls it accepts, for an Op whose attribute
    decides whether its result is a view or a new value.
    """
    buffer_alias_registry.register(op_cls, (param, when))


def aliased_operand(call: Call) -> int | None:
    """Return the aliased input's argument position, or None for a new value."""
    declared = buffer_alias_registry.lookup(type(call.target))
    if declared is None:
        return None
    param, when = declared
    if when is not None and not when(call):
        return None
    inputs = tuple(p for p in type(call.target)._op_schema.signature if p.kind == "input")
    return inputs.index(param)


def _addressing(layout) -> "tuple | None":
    """An addressable layout as (swizzle, base offset, outer Layout), or None.

    A shard layout that every participant reads whole (all ``Broadcast``) addresses
    the bytes its own layout does.
    """
    while isinstance(layout, ShardLayout) and all(
        isinstance(attr, Broadcast) for attr in layout.attrs
    ):
        layout = layout.layout
    if isinstance(layout, Layout) and not isinstance(layout, ShardLayout):
        return None, 0, layout
    if (
        isinstance(layout, ComposedLayout)
        and isinstance(layout.outer, Layout)
        and not isinstance(layout.outer, ShardLayout)
    ):
        return layout.inner, layout.offset, layout.outer
    return None


def _element_offset(coordinate: tuple, layout: Layout) -> int:
    """Where one logical coordinate lands: each axis read row-major through its modes."""
    total = 0
    for value, extents, steps in zip(coordinate, layout.shape, layout.strides, strict=True):
        extents = tuple(flatten(extents)) if isinstance(extents, tuple) else (extents,)
        steps = tuple(flatten(steps)) if isinstance(steps, tuple) else (steps,)
        for extent, step in zip(reversed(extents), reversed(steps), strict=True):
            total += (value % extent) * step
            value //= extent
    return total


def same_addresses(shape: tuple, first, second) -> bool:
    """Whether two layouts of one shape put every coordinate at the same address.

    Only a static, addressable pair is compared: same swizzle, same base offset, and
    the same offset for each coordinate however the modes are grouped.
    """
    left, right = _addressing(first), _addressing(second)
    if left is None or right is None or left[:2] != right[:2]:
        return False
    sizes = tuple(static_dim_value(extent) for extent in shape)
    if None in sizes:
        return False
    for _inner, _offset, outer in (left, right):
        steps = tuple(flatten(outer.strides)) if outer.strides is not None else (None,)
        if len(outer.shape) != len(sizes) or any(not isinstance(step, int) for step in steps):
            return False
    return all(
        _element_offset(coordinate, left[2]) == _element_offset(coordinate, right[2])
        for coordinate in itertools.product(*(range(size) for size in sizes))
    )
