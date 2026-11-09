"""Layout checks of values that need no registered rule."""

from __future__ import annotations

from tilefoundry.ir import core, expr, types
from tilefoundry.ir import utils as ir_utils


def require_layout(layout: object, node: expr.Expr) -> None:
    """Require *layout*, and every layout inside it, to be stated on *node*."""
    if layout is None:
        where = expr.diagnostic_location(node)
        raise expr.VerifyError(
            f"value {getattr(node, 'name', type(node).__name__)!r} has no tensor layout"
            + (f"\n  at {where}" if where else "")
        )
    if isinstance(layout, types.ShardLayout):
        require_layout(layout.layout, node)
    elif isinstance(layout, types.ComposedLayout):
        for child in (layout.outer, layout.inner):
            if isinstance(child, types.LayoutBase):
                require_layout(child, node)


def require_layouts(value: expr.Expr) -> None:
    """Require concrete tensor layouts on values, independently of annotations."""
    for node in (value, *core.collect_exprs(value)):
        bound = node.params if isinstance(node, (core.LoopRegion, core.MeshRegion)) else ()
        if isinstance(node, core.LoopRegion):
            bound = (*bound, node.induction_var)
        for held in (node, *bound):
            for tensor in ir_utils.tensor_types(held.type):
                require_layout(tensor.layout, held)


__all__ = ["require_layouts"]
