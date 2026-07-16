from __future__ import annotations

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
    """
    if names is None:
        names = ("g",) if len(layout_shape) == 1 else tuple("abcdef"[: len(layout_shape)])
    return Mesh(topology=topology, layout=tuple(layout_shape), names=tuple(names), topologies=(topology,))
