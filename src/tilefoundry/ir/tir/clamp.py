"""Effect-ful TIR Clamp Op — element-wise per-thread bound."""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types

IN_RMEM = pattern.tensor_in(expr.StorageKind.RMEM)


@expr.register_op(dialect="T", category="arith")
class Clamp(expr.Op):
    """Per-thread element-wise clamp: dst(i) = min(max(src(i), min_val), max_val)."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    min_val = expr.ParamDef(kind="attribute", annotation=float)
    max_val = expr.ParamDef(kind="attribute", annotation=float)
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=IN_RMEM)

    between = (pattern.SameConstraint("dtype", "src", "dst"),)


@typeinfer.register(Clamp)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Clamp)(access.identity_relations)


__all__ = ["Clamp"]
