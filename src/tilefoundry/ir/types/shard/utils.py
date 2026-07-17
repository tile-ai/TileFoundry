from __future__ import annotations

from .layout import ComposedLayout, Layout
from .mesh import Mesh, Topology


def _as_layout(mesh: Mesh) -> Layout:
    layout = mesh.layout
    if isinstance(layout, ComposedLayout):
        layout = layout.outer
    return Layout(shape=tuple(layout.shape), strides=tuple(layout.strides))


def _topology_domain(mesh: Mesh) -> int | None:
    """Return the selected mesh domain, or None for a dynamic extent."""
    extents = mesh.shape if isinstance(mesh.layout, ComposedLayout) else tuple(
        topology.size for topology in (mesh.topologies or (mesh.topology,))
    )
    domain = 1
    for extent in extents:
        if not isinstance(extent, int):
            return None
        domain *= extent
    return domain


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
