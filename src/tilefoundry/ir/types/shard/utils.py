from __future__ import annotations

import math

from .mesh import Mesh, Topology


def make_mesh(
    layout_shape: tuple,
    names: "tuple[str, ...] | None" = None,
    topology: "str | Topology" = "gpu",
) -> Mesh:
    """Convenience constructor for a ``Mesh`` with the given (logical) axis
    extents (C-order strides, via ``Mesh``'s own tuple-shorthand coercion).
    ``names`` defaults to ``a, b, c, ...`` (or ``g`` for a single axis) so a
    caller states only the extents instead of hand-building a ``Mesh``.

    ``topology`` accepts an explicit ``Topology`` or the ``"gpu"``-shorthand
    default; a raw string is resolved here into a real ``Topology`` sized to
    the domain (``Mesh`` itself rejects a string — ``docs/spec/shard.md``
    §5).
    """
    if names is None:
        names = ("g",) if len(layout_shape) == 1 else tuple("abcdef"[: len(layout_shape)])
    if isinstance(topology, str):
        topology = Topology(topology, math.prod(layout_shape))
    return Mesh(topology=topology, layout=tuple(layout_shape), names=tuple(names), topologies=(topology,))
