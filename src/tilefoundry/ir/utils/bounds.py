"""Concrete upper bounds of dimensions across the dispatch envelope."""

from __future__ import annotations

from tilefoundry.ir import expr, types

__all__ = ["register_strides", "shape_numel_upper_bound", "shape_upper_bound", "upper_bound"]


def upper_bound(dim) -> int:
    """Return a concrete int upper-bound element count for ``dim``."""
    if isinstance(dim, expr.DimVar):
        return int(dim.hi)
    static = expr.static_dim_value(dim)
    if static is not None:
        return static
    return int(dim)


def shape_numel_upper_bound(shape) -> int:
    """Product of per-dim upper bounds.

    The static element count a buffer or layout must hold across every runtime
    shape in the dispatch envelope.
    """
    n = 1
    for s in shape:
        n *= upper_bound(s)
    return n


def shape_upper_bound(shape) -> tuple[int, ...]:
    """Map ``upper_bound`` over every entry of *shape*."""
    return tuple(upper_bound(s) for s in shape)


def register_strides(sl: types.ShardLayout) -> tuple[int, ...]:
    """``sl``'s strides as steps on a *register* engine, not on a shared buffer.

    Registers are the distinct-engine-per-instance case of [shard
    §7.1.2](docs/spec/shard.md#712-layoutstrides): a ``Split(k)`` axis gets
    ``0``, so the projection offsets no instance along it, and the rest get
    the array's own compact steps, assigned from smallest to largest stated
    physical stride. This preserves register numbering when modes are reordered.
    An extent-1 axis names no step.
    """
    local = types.shard_layout_local_shape(sl, require_static=False)
    split_axes = {int(a.axis) for a in sl.attrs if isinstance(a, types.Split)}
    physical = types.flatten(sl.layout)
    order = sorted(range(len(local)), key=lambda axis: int(upper_bound(physical.strides[axis])))
    strides: list[int] = [0] * len(local)
    step = 1
    for axis in order:
        width = int(upper_bound(local[axis]))
        if axis not in split_axes and width != 1:
            strides[axis] = step
            step *= width
    return tuple(strides)
