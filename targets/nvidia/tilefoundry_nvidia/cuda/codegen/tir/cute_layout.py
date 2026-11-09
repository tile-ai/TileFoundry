"""One recursive renderer for a CuTe layout, in its type and value forms.

A layout is an int-tuple shape and an int-tuple stride; sub-tuples nest
``cute::Shape`` / ``cute::make_shape`` with them, recursively. Every emitter
spells a layout through these two functions, so a nested mode -- an operand
tile, a multi-level mesh -- renders everywhere a flat one does.
"""

from __future__ import annotations


def nest_types(values, tag):
    return ", ".join(
        f"{tag}<{nest_types(value, tag)}>" if isinstance(value, tuple) else f"cute::Int<{value}>"
        for value in values
    )


def nest_values(values, render, tag):
    return ", ".join(
        f"{tag}({nest_values(value, render, tag)})" if isinstance(value, tuple) else render(value)
        for value in values
    )


def render_layout_type(shape, stride) -> str:
    """The ``cute::Layout`` type of int-tuple *shape* and *stride*."""
    return (
        f"cute::Layout<cute::Shape<{nest_types(shape, 'cute::Shape')}>, "
        f"cute::Stride<{nest_types(stride, 'cute::Stride')}>>"
    )


def render_layout_value(shape, stride, dim, stride_dim) -> str:
    """The ``cute::make_layout`` value of int-tuple *shape* and *stride*.

    *dim* and *stride_dim* render one shape entry and one stride entry, which
    is where a runtime-provided extent reaches the emitted layout.
    """
    return (
        f"cute::make_layout(cute::make_shape({nest_values(shape, dim, 'cute::make_shape')}), "
        f"cute::make_stride({nest_values(stride, stride_dim, 'cute::make_stride')}))"
    )


__all__ = ["render_layout_type", "render_layout_value"]
