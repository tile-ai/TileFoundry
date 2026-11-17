"""Recover lexical mesh selections for source inspection."""

from __future__ import annotations

from tilefoundry.ir import types
from tilefoundry.ir.types import layout_algebra


def selection_layout(mesh: types.Mesh, selection: types.Mesh) -> types.Layout | None:
    """Invert reversed selection modes so local indices use row-major numbering.

    Composition uses AssertionError to reject indivisible mode strides.
    """
    if mesh.topologies != selection.topologies or len(mesh.topologies) != 1:
        return None
    if not isinstance(mesh.layout, types.ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, types.Layout) or mesh.layout.offset != selection.layout(0):
        return None
    source = types.flatten(types.plain_layout(selection.layout))
    target = types.flatten(mesh.layout.outer)
    if types.size(source) != types.size(target):
        return None
    steps = source.strides
    if not all(isinstance(one, int) for one in (*source.shape, *steps, *target.shape, *target.strides)):
        return None
    reversed_source = types.Layout(tuple(reversed(source.shape)), tuple(reversed(steps)))
    try:
        inverse = layout_algebra.left_inverse(reversed_source, major="col")
        local = layout_algebra.composition(inverse, target, major="col")
        shape, steps = types.flatten_to_tuple(local.shape), types.flatten_to_tuple(local.strides)
        order = types.compact_row_major(shape)
        local = types.Layout(shape, tuple(
            order[axis] if extent == 1 else step
            for axis, (extent, step) in enumerate(zip(shape, steps))
        ))
        if layout_algebra.composition(source, local, major="row") != target:
            return None
    except (AssertionError, TypeError, ValueError):
        return None
    return local


__all__ = ["selection_layout"]
