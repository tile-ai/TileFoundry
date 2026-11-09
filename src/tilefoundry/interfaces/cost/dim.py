"""Cost of scalar index arithmetic on symbolic dimensions.

The ``Dim*`` nodes are defined below the interfaces layer, so their evaluators
are registered here rather than in an Op module.
"""

from __future__ import annotations

from tilefoundry.ir import expr

from .context import Context, Cost
from .helpers import idle
from .registry import register

__all__: list[str] = []


@register(expr.DimAdd)
@register(expr.DimSub)
@register(expr.DimMul)
@register(expr.DimFloorDiv)
@register(expr.DimMod)
@register(expr.DimMin)
@register(expr.DimMax)
def dim_arithmetic(call: expr.Call, ctx: Context) -> Cost:
    """Scalar index arithmetic is address work, not tensor work: it costs nothing."""
    return Cost({}, idle(call))
