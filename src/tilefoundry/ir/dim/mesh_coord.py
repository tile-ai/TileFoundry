"""The coordinate of the unit a region is running on."""

from __future__ import annotations

from tilefoundry.ir import expr, types

__all__ = ["MeshCoord", "literal_axes"]


@expr.register_op(dialect="tf", category="sharding")
class MeshCoord(expr.Op):
    """This unit's coordinates along mesh axes *axes* of *mesh*: a scalar gives one, a vector many.

    Naming a mesh axis and asking which unit this is are two different
    questions. The first fills in a ``Split`` on a layout and leaves no node
    behind; only the second needs an operation, and this is it. So the mesh is a
    value the node carries rather than the scope it sits in, and whether an
    enclosing region binds that mesh stays a rule about the regions. The axes
    are an operand rather than a name so they can be computed, and so the node
    reads like every other Op in this dialect.
    """

    mesh = expr.ParamDef(kind="attribute", annotation=types.Mesh)
    axes = expr.ParamDef(kind="input")


def literal_axes(mesh_axes) -> tuple[int, ...] | None:
    """The mesh axes a literal operand names, or None when any of them is computed."""
    if isinstance(mesh_axes, expr.Tuple):
        found = tuple(expr.static_dim_value(mesh_axis) for mesh_axis in mesh_axes.elements)
        return None if None in found else found
    mesh_axis = expr.static_dim_value(mesh_axes)
    if mesh_axis is not None:
        return (mesh_axis,)
    if isinstance(mesh_axes, expr.Constant) and isinstance(mesh_axes.value, (list, tuple)):
        return tuple(int(mesh_axis) for mesh_axis in mesh_axes.value)
    return None
