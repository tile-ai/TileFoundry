"""Cost of ``MeshCoord``: which unit this is costs nothing."""

from __future__ import annotations

from tilefoundry.ir import dim, expr

from .context import Context, Cost
from .helpers import idle
from .registry import register

__all__: list[str] = []


@register(dim.MeshCoord)
def cost_mesh_coord(call: expr.Call, ctx: Context) -> Cost:
    """Which unit this is costs nothing: the machine already knows."""
    return Cost({}, idle(call))
