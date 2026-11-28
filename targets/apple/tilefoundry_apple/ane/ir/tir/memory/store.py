"""dst = src, written from the L2 buffer out to unified memory."""

from tilefoundry.ir import expr, pattern
from tilefoundry.ir.tir import memory
from tilefoundry_apple.ane.ir.tir import layout


@memory.register_copy_semantics
@expr.register_op(dialect="T", category="memory", namespace="ane", name="store")
class Store(expr.Op):
    """dst = src: an activation in the L2 buffer copied out to unified memory."""

    capability = expr.OpCapability("memory.store")
    execution_mesh = layout.ENGINE
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.activation("src", layout.SMEM_ONLY))
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.activation("dst", layout.GMEM_ONLY))
    between = (pattern.SameConstraint("shape", "src", "dst"),)


__all__ = ["Store"]
