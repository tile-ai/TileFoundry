"""Effect-ful TIR Op ``tir.nn.ReLU``.

Tensor-level pointwise ReLU — writes element-wise ``max(src, 0)``
into ``dst`` (in-place memory write). Wrapped by
``Evaluate(ReLU, ...)`` in Stmt position.
"""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types

IN_RMEM = pattern.tensor_in(expr.StorageKind.RMEM)


@expr.register_op(dialect='T', category='nn')
class ReLU(expr.Op):
    """Tensor-level pointwise ReLU writing into ``dst``."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=IN_RMEM)

    between = (
        pattern.SameConstraint("shape", "src", "dst"),
        pattern.SameConstraint("dtype", "src", "dst"),
    )


@typeinfer.register(ReLU)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(ReLU)(access.identity_relations)


__all__ = ["ReLU"]
