"""The coordinate of the unit a region is running on."""

from __future__ import annotations

from tilefoundry.ir import expr, types

__all__ = ["MeshCoord", "literal_axes"]


@expr.register_op(dialect="tf", category="sharding")
class MeshCoord(expr.Op):
    """This unit's coordinates along *axes* of *mesh*: a scalar axis gives one, a vector one each.

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


def literal_axes(axes) -> tuple[int, ...] | None:
    """The axes a literal operand names, or None when any of them is computed."""
    if isinstance(axes, expr.Tuple):
        found = tuple(expr.static_dim_value(axis) for axis in axes.elements)
        return None if None in found else found
    axis = expr.static_dim_value(axes)
    if axis is not None:
        return (axis,)
    if isinstance(axes, expr.Constant) and isinstance(axes.value, (list, tuple)):
        return tuple(int(axis) for axis in axes.value)
    return None
