"""One recursive renderer for a CuTe layout, in its type and value forms.

A layout is an int-tuple shape and an int-tuple stride; sub-tuples nest
``cute::Shape`` / ``cute::make_shape`` with them, recursively. Every emitter
spells a layout through these two functions, so a nested mode -- an operand
tile, a multi-level mesh -- renders everywhere a flat one does.
"""

from __future__ import annotations


def render_layout_type(shape, stride) -> str:
    """The ``cute::Layout`` type of int-tuple *shape* and *stride*."""

    def nest(values, tag):
        return ", ".join(
            f"{tag}<{nest(value, tag)}>" if isinstance(value, tuple) else f"cute::Int<{value}>"
            for value in values
        )

    return (
        f"cute::Layout<cute::Shape<{nest(shape, 'cute::Shape')}>, "
        f"cute::Stride<{nest(stride, 'cute::Stride')}>>"
    )


def render_layout_value(shape, stride, dim, stride_dim) -> str:
    """The ``cute::make_layout`` value of int-tuple *shape* and *stride*.

    *dim* and *stride_dim* render one shape entry and one stride entry, which
    is where a runtime-provided extent reaches the emitted layout.
    """

    def nest(values, render, tag):
        return ", ".join(
            f"{tag}({nest(value, render, tag)})" if isinstance(value, tuple) else render(value)
            for value in values
        )

    return (
        f"cute::make_layout(cute::make_shape({nest(shape, dim, 'cute::make_shape')}), "
        f"cute::make_stride({nest(stride, stride_dim, 'cute::make_stride')}))"
    )
