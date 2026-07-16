"""Match an enclosing mesh scope against an op's required thread scope."""
from __future__ import annotations

from .layout import ComposedLayout, Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh


def _as_layout(mesh: Mesh) -> Layout:
    layout = mesh.layout
    if isinstance(layout, ComposedLayout):
        layout = layout.outer
    return Layout(shape=tuple(layout.shape), strides=tuple(layout.strides))


def _topology_domain(mesh: Mesh) -> "int | None":
    """Total thread count = product of the mesh's topology extents; ``None`` if
    any extent is dynamic (launch-provided)."""
    if isinstance(mesh.layout, ComposedLayout):
        extents = mesh.shape
    else:
        extents = tuple(t.size for t in (mesh.topologies or (mesh.topology,)))
    domain = 1
    for extent in extents:
        if not isinstance(extent, int):
            return None
        domain *= extent
    return domain


def mesh_scope_matches_required_scope(current: Mesh, required: Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs."""
    # Same program topology level — a `cta` scope is never a `thread`/warp scope.
    if current.topology.name != required.topology.name:
        return False

    cur_domain = _topology_domain(current)
    req_domain = _topology_domain(required)
    if cur_domain is None or req_domain is None:
        return False

    cur_layout = _as_layout(current)
    req_layout = _as_layout(required)

    # Self-consistent mesh: topology domain == layout extent (Mesh does not
    # enforce this, so a `thread(64)` carrying a 32-element layout is rejected).
    if cur_domain != size(cur_layout) or req_domain != size(req_layout):
        return False

    # Must be an admissible execution scope (injective, compact-ordered).
    if not is_inverse_projectable(cur_layout):
        return False

    # Same thread-value decomposition: the fragment's Split attrs index the mesh
    # axes, so the lane layout must match exactly (shape + strides) — a flat or
    # differently-shaped 32-lane scope cannot host a 2-axis (4, 8) fragment.
    return cur_layout.shape == req_layout.shape and cur_layout.strides == req_layout.strides


__all__ = ["mesh_scope_matches_required_scope"]
