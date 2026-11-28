"""dst = src, read from unified memory into the L2 buffer."""

from tilefoundry.ir import expr
from tilefoundry_apple.ane.ir.tir import layout

from .transfer import register_transfer


@register_transfer
@expr.register_op(dialect="T", category="memory", namespace="ane", name="load")
class Load(expr.Op):
    """dst = src: an activation in unified memory copied into the L2 buffer."""

    capability = expr.OpCapability("memory.load")
    execution_mesh = layout.ENGINE
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.activation("src", layout.GMEM_ONLY))
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.activation("dst", layout.SMEM_ONLY))
    between = (layout.SameFeatureConstraint("shape", "src", "dst"),)


__all__ = ["Load"]
