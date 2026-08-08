from __future__ import annotations

import math

from .layout import Layout
from .layout_algebra import try_c_order_strides
from .mesh import Mesh, Topology


def make_mesh(
    layout_shape: tuple,
    names: "tuple[str, ...] | None" = None,
    topology: "str | Topology" = "gpu",
) -> Mesh:
    """Convenience constructor for a ``Mesh`` with the given (logical) axis
    extents and C-order strides. ``names`` defaults to ``a, b, c, ...`` (or
    ``g`` for a single axis) so a caller states only the extents instead of
    hand-building a ``Mesh``.

    ``topology`` accepts an explicit ``Topology`` or the ``"gpu"``-shorthand
    default; a raw string is resolved here into a real ``Topology`` sized to
    the domain.
    """
    if names is None:
        names = ("g",) if len(layout_shape) == 1 else tuple("abcdef"[: len(layout_shape)])
    if isinstance(topology, str):
        topology = Topology(topology, math.prod(layout_shape))
    layout_shape = tuple(layout_shape)
    return Mesh(
        topologies=(topology,),
        layout=Layout(shape=layout_shape, strides=try_c_order_strides(layout_shape)),
        names=tuple(names),
    )
