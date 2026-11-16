"""Which scope a program runs in, and what that scope admits.

Entering a `with Mesh(...)` composes the scope in force with the one it names;
a value is read where the units it is laid out over are the units the
scope runs on. Both questions are about the mesh a statement stands inside,
which is neither a type nor a visitor, so they live here rather than with
either.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir import expr, types


def flat_layout(mesh: types.Mesh) -> types.Layout | types.ComposedLayout:
    """The whole mesh as one layout, offset included: the form a mesh is written and printed in.

    The index runs across all topologies of the mesh. The reverse of ``nested_layout``.
    A sliced mesh keeps its ``ComposedLayout``, a zero offset included.
    """
    outer = types.plain_layout(mesh.layout)
    if outer is None:
        raise ValueError(
            "a mesh whose slice states an identity box states no layout of "
            "its own, so its levels select nothing"
        )
    flat = types.flatten(outer)
    return replace(mesh.layout, outer=flat) if isinstance(mesh.layout, types.ComposedLayout) else flat


def storage_reaches(storage, mesh: types.Mesh, current: types.Mesh) -> bool:
    """Whether *storage* reaches across a coarser value-to-scope boundary."""
    if current.topologies[-1].name in types.topology_names(mesh):
        return True
    try:
        storage = expr.resolve_storage(storage)
    except (TypeError, ValueError):
        return False
    return storage in {expr.StorageKind.GMEM, expr.StorageKind.SMEM}


__all__ = ["flat_layout", "storage_reaches"]
