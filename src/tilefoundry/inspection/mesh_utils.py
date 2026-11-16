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
    if not isinstance(mesh.layout.outer, types.Layout) or mesh.layout.offset != types.offset(selection):
        return None
    source = types.flatten(types.layouts(selection)[0])
    target = types.flatten(mesh.layout.outer)
    if types.size(source) != types.size(target):
        return None
    steps = source.strides
    if not all(isinstance(one, int) for one in (*source.shape, *steps, *target.shape, *target.strides)):
        return None
    reversed_source = types.Layout(tuple(reversed(source.shape)), tuple(reversed(steps)))
    try:
        local = types.flatten(layout_algebra.composition(layout_algebra.left_inverse(reversed_source), target))
        order = types.compact_row_major(local.shape)
        local = types.Layout(local.shape, tuple(
            order[axis] if extent == 1 else step
            for axis, (extent, step) in enumerate(zip(local.shape, local.strides))
        ))
        if layout_algebra.composition(source, local, major="row") != target:
            return None
    except (AssertionError, TypeError, ValueError):
        return None
    return local


__all__ = ["selection_layout"]
