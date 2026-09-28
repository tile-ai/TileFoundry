"""The arrangement a tensor value presents to an immediate consumer."""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.types import Layout, TensorType
from tilefoundry.ir.types.stride import try_compact_major


def presented_layout_of(value, ctx):
    """Return the value's presented arrangement without changing its stored type.

    A view operation owns how it presents its source.  An ordinary tensor whose
    type leaves layout unstated presents the compact arrangement all HIR readers
    assign to that value.
    """
    type_ = ctx.type_of(value)
    if not isinstance(type_, TensorType):
        return None
    if type_.layout is not None:
        return type_.layout
    if isinstance(value, Call):
        query = getattr(value.target, "presented_layout", None)
        if query is not None:
            return query(value, ctx)
    strides = try_compact_major(tuple(type_.shape))
    return None if strides is None else Layout(tuple(type_.shape), strides)


__all__ = ["presented_layout_of"]
